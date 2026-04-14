"""
Extract DETR cross-attention embeddings for existing oracle samples.

For each of the 1368 training samples, runs:
  backbone (once) → FPN(L32) + DETR decoder → mean-pool last decoder layer
  → (256,) cross-modal fusion embedding

This is SAM3's OWN text-image cross-attention output — far richer than
concat(text_emb, img_emb) because SAM3 has already aligned text concepts
with image features inside its DETR decoder.

Shape saved: detr_embs.npy  (1368, 256)

Usage:
    cd capr_clean
    python experiments/extract_detr_embs.py
"""
import os, sys, json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam3_wrapper import SAM3Wrapper

DATA_FILE  = ("/home/grads/f/fahimehorvatinia/Documents/newpaper_2026"
              "/saco_gold_data/metaclip/saco_gold_metaclip_test_1.json")
IMAGE_ROOT = "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/metaclip-images"
OUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "router_training_data")


def main():
    print("\n=== CAPR v3 — Extracting DETR Cross-Attention Embeddings ===\n")
    print("Router input: SAM3's own text-image cross-attention output (256-dim)")
    print("Method: FPN(L32) + DETR decoder → last layer mean over 200 queries\n")

    meta_path = os.path.join(OUT_DIR, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    samples = meta["samples"]
    N = len(samples)
    print(f"Samples: {N}")

    with open(DATA_FILE) as f:
        data = json.load(f)
    id_to_info = {img["id"]: {"path": os.path.join(IMAGE_ROOT, img["file_name"]),
                               "prompt": img["text_input"]}
                  for img in data["images"]}

    wrapper  = SAM3Wrapper()
    detr_embs = []
    skipped   = 0

    for s in tqdm(samples, desc="Extracting DETR embs"):
        info = id_to_info.get(s["image_id"])
        if info is None or not os.path.exists(info["path"]):
            detr_embs.append(np.zeros(256, dtype=np.float32))
            skipped += 1
            continue
        try:
            img = Image.open(info["path"]).convert("RGB")
            pv, ids = wrapper.preprocess(img, info["prompt"])
            hs, backbone_lhs, text_emb_detr, _, _ = wrapper.extract(pv, ids)
            # Run FPN(L32) + DETR → last decoder cross-attn output (256,)
            emb = wrapper.extract_detr_emb(hs, backbone_lhs, text_emb_detr, pv, layer_idx=32)
            detr_embs.append(emb.cpu().numpy())
        except Exception as e:
            detr_embs.append(np.zeros(256, dtype=np.float32))
            skipped += 1

    detr_embs = np.array(detr_embs, dtype=np.float32)
    out_path  = os.path.join(OUT_DIR, "detr_embs.npy")
    np.save(out_path, detr_embs)

    print(f"\nDone.  Extracted: {N-skipped}  skipped/zeros: {skipped}")
    print(f"Saved: {out_path}  shape={detr_embs.shape}")


if __name__ == "__main__":
    main()
