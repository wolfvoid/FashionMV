"""Validation datasets for FashionMV CIR evaluation."""

import json
import os
from pathlib import Path
from typing import List, Optional, Set

from PIL import Image
from torch.utils.data import Dataset

MAX_VIEWS = 5

VALID_DATASETS = {"deepfashion", "f200k", "fashiongen_val"}

DATASET_IMG_SUBDIR = {
    "deepfashion": "deepfashion",
    "f200k": "f200k",
    "fashiongen_val": "fashiongen_val",
}


class CIRValDataset(Dataset):
    """Validation CIR triplets with images.

    Args:
        data_dir: Directory containing val_triplets.jsonl.
        image_root: Root directory with dataset image folders.
        datasets: If provided, only load triplets from these datasets.
                  Valid values: "deepfashion", "f200k", "fashiongen_val".
    """

    def __init__(self, data_dir: str, image_root: str,
                 datasets: Optional[Set[str]] = None):
        self.image_root = image_root
        self.samples = []

        triplet_path = os.path.join(data_dir, "val_triplets.jsonl")
        with open(triplet_path) as f:
            for line in f:
                item = json.loads(line)
                ds = item["dataset"]

                if datasets and ds not in datasets:
                    continue

                img_subdir = DATASET_IMG_SUBDIR.get(ds, ds)
                src_dir = os.path.join(image_root, img_subdir, str(item["source_id"]))
                tgt_dir = os.path.join(image_root, img_subdir, str(item["target_id"]))
                if not os.path.isdir(src_dir) or not os.path.isdir(tgt_dir):
                    continue

                src_imgs = self._list_images(src_dir)[:MAX_VIEWS]
                tgt_imgs = self._list_images(tgt_dir)[:MAX_VIEWS]
                if not src_imgs or not tgt_imgs:
                    continue

                self.samples.append({
                    "source_id": str(item["source_id"]),
                    "target_id": str(item["target_id"]),
                    "dataset": ds,
                    "source_image_paths": src_imgs,
                    "target_image_paths": tgt_imgs,
                    "modification_text_short": item["modification_text_short"],
                })

        ds_counts = {}
        for s in self.samples:
            ds_counts[s["dataset"]] = ds_counts.get(s["dataset"], 0) + 1
        ds_str = ", ".join(f"{k}: {v}" for k, v in sorted(ds_counts.items()))
        print(f"[CIRValDataset] Loaded {len(self.samples)} triplets ({ds_str})")

    @staticmethod
    def _list_images(img_dir: str) -> List[str]:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        return sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if Path(f).suffix.lower() in exts
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        src_imgs = [Image.open(p).convert("RGB") for p in s["source_image_paths"]]
        tgt_imgs = [Image.open(p).convert("RGB") for p in s["target_image_paths"]]
        return {
            "source_id": s["source_id"],
            "target_id": s["target_id"],
            "dataset": s["dataset"],
            "source_images": src_imgs,
            "target_images": tgt_imgs,
            "modification_text_short": s["modification_text_short"],
        }


class ProductValDataset(Dataset):
    """All validation products for building the retrieval gallery.

    Args:
        data_dir: Directory containing val_triplets.jsonl.
        image_root: Root directory with dataset image folders.
        datasets: If provided, only load products from these datasets.
    """

    def __init__(self, data_dir: str, image_root: str,
                 datasets: Optional[Set[str]] = None):
        self.samples = []

        triplet_path = os.path.join(data_dir, "val_triplets.jsonl")
        product_ids = set()
        dataset_map = {}
        with open(triplet_path) as f:
            for line in f:
                item = json.loads(line)
                if datasets and item["dataset"] not in datasets:
                    continue
                for pid in [str(item["source_id"]), str(item["target_id"])]:
                    if pid not in product_ids:
                        product_ids.add(pid)
                        dataset_map[pid] = item["dataset"]

        for pid in product_ids:
            ds = dataset_map[pid]
            img_subdir = DATASET_IMG_SUBDIR.get(ds, ds)
            img_dir = os.path.join(image_root, img_subdir, pid)
            if not os.path.isdir(img_dir):
                continue
            imgs = CIRValDataset._list_images(img_dir)[:MAX_VIEWS]
            if not imgs:
                continue
            self.samples.append({
                "product_id": pid,
                "dataset": ds,
                "image_paths": imgs,
            })

        print(f"[ProductValDataset] Loaded {len(self.samples)} products")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        imgs = [Image.open(p).convert("RGB") for p in s["image_paths"]]
        return {
            "product_id": s["product_id"],
            "dataset": s["dataset"],
            "images": imgs,
        }
