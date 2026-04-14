import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor


class FashionEmbeddingModel(nn.Module):
    """Wraps Qwen3.5 as an embedding extractor using <emb_all> token."""

    def __init__(self, model_path: str, emb_all_token_id: int, processor: AutoProcessor):
        super().__init__()
        self.vlm = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
        )
        self.emb_all_token_id = emb_all_token_id
        self.processor = processor

    def _find_emb_all_positions(self, input_ids: torch.Tensor) -> List[int]:
        ids = input_ids.view(-1)
        pos = (ids == self.emb_all_token_id).nonzero(as_tuple=True)[0]
        return [p.item() for p in pos]

    def _forward_core(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        self.vlm.model.rope_deltas = None
        kwargs = {}
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            kwargs["image_grid_thw"] = image_grid_thw
        if mm_token_type_ids is not None:
            kwargs["mm_token_type_ids"] = mm_token_type_ids

        outputs = self.vlm.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs,
        )
        self.vlm.model.rope_deltas = None
        last_hidden = outputs.last_hidden_state

        B = input_ids.shape[0]
        all_positions = []
        for i in range(B):
            all_positions.append(self._find_emb_all_positions(input_ids[i]))
        return last_hidden, all_positions

    def _pad_and_batch(self, inputs_list: List[Dict], device: torch.device):
        all_ids = [inp["input_ids"].squeeze(0).to(device) for inp in inputs_list]
        max_len = max(ids.shape[0] for ids in all_ids)

        batched_ids, batched_mask = [], []
        for ids in all_ids:
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                batched_ids.append(torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype, device=device)]))
                batched_mask.append(torch.cat([
                    torch.ones(ids.shape[0], dtype=torch.long, device=device),
                    torch.zeros(pad_len, dtype=torch.long, device=device),
                ]))
            else:
                batched_ids.append(ids)
                batched_mask.append(torch.ones(ids.shape[0], dtype=torch.long, device=device))

        batched_ids = torch.stack(batched_ids)
        batched_mask = torch.stack(batched_mask)

        cat_pv = torch.cat([inp["pixel_values"].to(device, dtype=torch.bfloat16) for inp in inputs_list], dim=0)
        cat_thw = torch.cat([inp["image_grid_thw"].to(device) for inp in inputs_list], dim=0)

        mm_token_type_ids = None
        if inputs_list[0].get("mm_token_type_ids") is not None:
            mm_list = []
            for inp in inputs_list:
                mm = inp["mm_token_type_ids"].squeeze(0).to(device)
                pad_len = max_len - mm.shape[0]
                if pad_len > 0:
                    mm = torch.cat([mm, torch.zeros(pad_len, dtype=mm.dtype, device=device)])
                mm_list.append(mm)
            mm_token_type_ids = torch.stack(mm_list)

        return batched_ids, batched_mask, cat_pv, cat_thw, mm_token_type_ids

    def forward_visual_batch(
        self, visual_inputs_list: List[Dict], device: torch.device
    ) -> List[torch.Tensor]:
        """Single-turn visual forward. Returns list of (H,) embeddings."""
        if not visual_inputs_list:
            return []
        ids, mask, pv, thw, mm = self._pad_and_batch(visual_inputs_list, device)
        last_hidden, all_pos = self._forward_core(ids, mask, pv, thw, mm)

        results = []
        for i in range(len(visual_inputs_list)):
            positions = all_pos[i]
            results.append(last_hidden[i, positions[-1]] if positions else last_hidden[i, -1])
        return results

    def forward_visual_batch_multiturn(
        self, visual_inputs_list: List[Dict], device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Multi-turn visual forward. Returns (s_list, q_list).
        s = first <emb_all> (source embedding), q = last <emb_all> (query embedding)."""
        if not visual_inputs_list:
            return [], []
        ids, mask, pv, thw, mm = self._pad_and_batch(visual_inputs_list, device)
        last_hidden, all_pos = self._forward_core(ids, mask, pv, thw, mm)

        s_list, q_list = [], []
        for i in range(len(visual_inputs_list)):
            positions = all_pos[i]
            if len(positions) >= 2:
                s_list.append(last_hidden[i, positions[0]])
                q_list.append(last_hidden[i, positions[-1]])
            elif len(positions) == 1:
                s_list.append(last_hidden[i, positions[0]])
                q_list.append(last_hidden[i, positions[0]])
            else:
                s_list.append(last_hidden[i, -1])
                q_list.append(last_hidden[i, -1])
        return s_list, q_list
