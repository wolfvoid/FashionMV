"""
Evaluate ProCIR on FashionMV validation set.

Each dataset is evaluated independently with its own gallery.
You can evaluate all three datasets at once, or select specific ones.

Usage:
  # Evaluate on all datasets
  python evaluate.py --model_path ./model --image_root ./images --data_dir ./data

  # Evaluate on a single dataset
  python evaluate.py --model_path ./model --image_root ./images --data_dir ./data \
    --datasets deepfashion

  # Evaluate on two datasets
  python evaluate.py --model_path ./model --image_root ./images --data_dir ./data \
    --datasets deepfashion f200k

  # Multi-GPU (DDP)
  torchrun --nproc_per_node=4 evaluate.py \
    --model_path ./model --image_root ./images --data_dir ./data
"""
import argparse
import json
import os
import time
from collections import OrderedDict, defaultdict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoProcessor

from procir import FashionEmbeddingModel
from procir.datasets import CIRValDataset, ProductValDataset, VALID_DATASETS
from procir.collators import DocCollator, CIRQueryCollator

EMB_TOKEN = "<emb_all>"

DS_DISPLAY = {
    "deepfashion": "DeepFashion",
    "f200k": "Fashion200K",
    "fashiongen_val": "FashionGen-val",
}


def compute_recall(q_embs, gallery_embs, gt_indices, ks=(1, 5, 10)):
    q_norm = F.normalize(q_embs.float(), dim=-1)
    g_norm = F.normalize(gallery_embs.float(), dim=-1)
    sim = q_norm @ g_norm.T
    gt_t = torch.tensor(gt_indices)
    results = {}
    for k in ks:
        _, topk = sim.topk(min(k, sim.shape[1]), dim=1)
        r = (topk == gt_t.unsqueeze(1)).any(dim=1).float().mean().item()
        results[f"R@{k}"] = round(r * 100, 2)
    return results


def setup_model(model_path, device):
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "right"
    tok = processor.tokenizer

    emb_token_id = tok.convert_tokens_to_ids(EMB_TOKEN)
    if emb_token_id == tok.unk_token_id or emb_token_id is None:
        tok.add_special_tokens({"additional_special_tokens": [EMB_TOKEN]})
        emb_token_id = tok.convert_tokens_to_ids(EMB_TOKEN)
        model = FashionEmbeddingModel(model_path, emb_token_id, processor)
        model.vlm.resize_token_embeddings(len(tok))
    else:
        model = FashionEmbeddingModel(model_path, emb_token_id, processor)

    model = model.to(device).eval()
    return model, processor, emb_token_id


def gather_tensors(local_tensor, world_size):
    if world_size == 1:
        return local_tensor
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_tensor)
    return torch.cat([g for g in gathered if g.shape[0] > 0], dim=0)


def gather_lists(local_list, world_size):
    if world_size == 1:
        return local_list
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_list)
    return sum(gathered, [])


def main():
    parser = argparse.ArgumentParser(description="Evaluate ProCIR on FashionMV")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image_root", type=str, required=True,
                        help="Root directory containing dataset image folders")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing val_triplets.jsonl")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--datasets", nargs="+", default=None,
                        choices=sorted(VALID_DATASETS),
                        help="Datasets to evaluate on. Default: all three. "
                             "Choices: deepfashion, f200k, fashiongen_val")
    args = parser.parse_args()

    selected_ds = set(args.datasets) if args.datasets else None
    ds_label = ", ".join(sorted(selected_ds)) if selected_ds else "all"

    # ── DDP setup ──
    use_ddp = "RANK" in os.environ
    if use_ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = rank == 0
    os.makedirs(args.output_dir, exist_ok=True)

    if is_main:
        print(f"{'='*60}")
        print(f"  FashionMV CIR Evaluation")
        print(f"  Model:    {args.model_path}")
        print(f"  Datasets: {ds_label}")
        print(f"  GPUs:     {world_size}")
        print(f"{'='*60}")

    model, processor, emb_token_id = setup_model(args.model_path, device)
    collator_args = (processor, emb_token_id)

    # ── Phase 1: Encode gallery products ──
    product_ds = ProductValDataset(args.data_dir, args.image_root, datasets=selected_ds)
    doc_collator = DocCollator(*collator_args)

    if use_ddp:
        sampler = DistributedSampler(product_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = SequentialSampler(product_ds)

    doc_loader = DataLoader(product_ds, batch_size=args.batch_size, sampler=sampler,
                            collate_fn=doc_collator, num_workers=2, pin_memory=False)

    local_doc_embs, local_doc_pids, local_doc_dsets = [], [], []
    t0 = time.time()

    with torch.no_grad():
        for batch in tqdm(doc_loader, desc="Encoding gallery", disable=not is_main):
            emb_list = model.forward_visual_batch(batch["doc_visual_inputs"], device)
            for j, emb in enumerate(emb_list):
                local_doc_embs.append(emb.cpu())
                local_doc_pids.append(batch["batch_meta"][j]["product_id"])
                local_doc_dsets.append(batch["batch_meta"][j]["dataset"])

    doc_t = torch.stack(local_doc_embs) if local_doc_embs else torch.zeros(0, 1024)
    all_doc_embs = gather_tensors(doc_t, world_size)
    all_doc_pids = gather_lists(local_doc_pids, world_size)
    all_doc_dsets = gather_lists(local_doc_dsets, world_size)

    doc_emb_dict = OrderedDict()
    doc_dset_dict = {}
    for i, pid in enumerate(all_doc_pids):
        if pid not in doc_emb_dict:
            doc_emb_dict[pid] = all_doc_embs[i]
            doc_dset_dict[pid] = all_doc_dsets[i]

    if is_main:
        print(f"Gallery: {len(doc_emb_dict)} products ({time.time()-t0:.1f}s)")

    # ── Phase 2: Encode CIR queries (two-stage dialogue) ──
    cir_ds = CIRValDataset(args.data_dir, args.image_root, datasets=selected_ds)
    query_collator = CIRQueryCollator(*collator_args)

    if use_ddp:
        cir_sampler = DistributedSampler(cir_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        cir_sampler = SequentialSampler(cir_ds)

    cir_loader = DataLoader(cir_ds, batch_size=args.batch_size, sampler=cir_sampler,
                            collate_fn=query_collator, num_workers=2, pin_memory=False)

    local_q_embs, local_s_embs = [], []
    local_tids, local_sids, local_q_dsets = [], [], []

    with torch.no_grad():
        for batch in tqdm(cir_loader, desc="Encoding queries", disable=not is_main):
            s_list, q_list = model.forward_visual_batch_multiturn(
                batch["query_visual_inputs"], device)
            for j in range(len(batch["batch_meta"])):
                local_q_embs.append(q_list[j].cpu())
                local_s_embs.append(s_list[j].cpu())
                local_tids.append(batch["batch_meta"][j]["target_id"])
                local_sids.append(batch["batch_meta"][j]["source_id"])
                local_q_dsets.append(batch["batch_meta"][j]["dataset"])

    q_t = torch.stack(local_q_embs) if local_q_embs else torch.zeros(0, 1024)
    s_t = torch.stack(local_s_embs) if local_s_embs else torch.zeros(0, 1024)
    all_q = gather_tensors(q_t, world_size)
    all_s = gather_tensors(s_t, world_size)
    all_tids = gather_lists(local_tids, world_size)
    all_sids = gather_lists(local_sids, world_size)
    all_q_dsets = gather_lists(local_q_dsets, world_size)

    for i, (sid, ds) in enumerate(zip(all_sids, all_q_dsets)):
        if sid not in doc_emb_dict:
            doc_emb_dict[sid] = all_s[i]
            doc_dset_dict[sid] = ds

    if is_main:
        print(f"Queries: {all_q.shape[0]}, Gallery after source reuse: {len(doc_emb_dict)}")

    # ── Phase 3: Compute Recall ──
    if is_main:
        q_groups = defaultdict(list)
        for i, ds in enumerate(all_q_dsets):
            q_groups[ds].append(i)

        results = {}
        print(f"\n{'='*60}")
        print(f"  CIR Results (short modification text)")
        print(f"{'='*60}")

        avg_r5, avg_r10 = [], []
        for ds_raw in sorted(q_groups.keys()):
            ds_name = DS_DISPLAY.get(ds_raw, ds_raw)
            q_idx = q_groups[ds_raw]

            pids = [p for p, d in doc_dset_dict.items() if d == ds_raw]
            gallery = torch.stack([doc_emb_dict[p] for p in pids])
            pid_to_idx = {p: i for i, p in enumerate(pids)}

            valid_q, gt = [], []
            for qi in q_idx:
                if all_tids[qi] in pid_to_idx:
                    valid_q.append(qi)
                    gt.append(pid_to_idx[all_tids[qi]])

            r = compute_recall(all_q[valid_q], gallery, gt)
            avg_r5.append(r["R@5"])
            avg_r10.append(r["R@10"])
            print(f"  {ds_name:15s}  R@1={r['R@1']:6.2f}  R@5={r['R@5']:6.2f}  R@10={r['R@10']:6.2f}  (queries={len(valid_q)}, gallery={len(pids)})")

            for k, v in r.items():
                results[f"{ds_name}_{k}"] = v

        print(f"  {'Average':15s}  R@5={sum(avg_r5)/len(avg_r5):6.2f}  R@10={sum(avg_r10)/len(avg_r10):6.2f}")
        results["Average_R@5"] = round(sum(avg_r5) / len(avg_r5), 2)
        results["Average_R@10"] = round(sum(avg_r10) / len(avg_r10), 2)

        result_path = os.path.join(args.output_dir, "eval_results.json")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {result_path}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
