"""Collators for batching visual inputs in CIR evaluation."""

from typing import Dict, List

from .chat_utils import patch_think_tokens

IMAGE_MIN_PIXELS = 128 * 128
IMAGE_MAX_PIXELS = 512 * 512


class BaseCollator:
    def __init__(self, processor, emb_token_id: int):
        self.processor = processor
        self.emb_token_id = emb_token_id

    def _ea(self):
        return self.processor.tokenizer.decode([self.emb_token_id])

    def _process_visual(self, text_str, images):
        return self.processor(
            text=[text_str], images=images, return_tensors="pt",
            min_pixels=IMAGE_MIN_PIXELS, max_pixels=IMAGE_MAX_PIXELS,
        )

    def _build_doc(self, images):
        ea = self._ea()
        content = [{"type": "image", "image": img} for img in images]
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": ea}]},
        ]
        t = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        t = patch_think_tokens(t)
        return self._process_visual(t, images)

    def _build_multiturn_query(self, images, mod_text):
        ea = self._ea()
        user1 = [{"type": "image", "image": img} for img in images]
        messages = [
            {"role": "user", "content": user1},
            {"role": "assistant", "content": [{"type": "text", "text": ea}]},
            {"role": "user", "content": [{"type": "text", "text": mod_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": ea}]},
        ]
        t = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        t = patch_think_tokens(t)
        return self._process_visual(t, images)


class DocCollator(BaseCollator):
    """Collate gallery products for single-turn visual encoding."""

    def __call__(self, batch: List[Dict]) -> Dict:
        doc_vis, meta = [], []
        for s in batch:
            doc_vis.append(self._build_doc(s["images"]))
            meta.append({"product_id": s["product_id"], "dataset": s["dataset"]})
        return {"doc_visual_inputs": doc_vis, "batch_meta": meta}


class CIRQueryCollator(BaseCollator):
    """Collate CIR queries for multi-turn visual encoding."""

    def __call__(self, batch: List[Dict]) -> Dict:
        query_vis, meta = [], []
        for s in batch:
            query_vis.append(self._build_multiturn_query(
                s["source_images"], s["modification_text_short"]))
            meta.append({
                "source_id": s["source_id"],
                "target_id": s["target_id"],
                "dataset": s["dataset"],
            })
        return {"query_visual_inputs": query_vis, "batch_meta": meta}
