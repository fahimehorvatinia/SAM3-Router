"""
Extract full per-query DETR sequences for the attention-based router (v4).

Unlike extract_detr_embs.py which mean-pools to (256,), this script saves the
complete last-layer DETR decoder output: (200, 256) per sample.

The attention router learns to weight individual queries rather than treating
all 200 equally — capturing that only a few queries correspond to the concept.

Shape saved:
    detr_embs_full.npy  (N, 200, 256)  float32

Usage:
    cd capr_clean
    python experiments/extract_detr_embs_full.py

    # For a specific domain's oracle data:
    DOMAIN=attributes python experiments/extract_detr_embs_full.py
"""
import os, sys, json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam3_wrapper import SAM3Wrapper

DOMAIN     = os.environ.get("DOMAIN", "metaclip")
OUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "router_training_data")

DATA_ROOTS = {
    "metaclip":          "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/saco_gold_data/metaclip",
    "attributes":        "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/saco_gold_data/attributes",
    "crowded":           "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/saco_gold_data/crowded",
    "wiki-food&drink":   "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/saco_gold_data/wiki-food&drink",
    "wiki-sports":       "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/saco_gold_data/wiki-sports_equipment",
    "wiki-common1k":     "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/saco_gold_data/wiki-common1k",
}
IMAGE_ROOT = "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/metaclip-images"


def main():
    print(f"\n=== CAPR v4 — Extracting Full DETR Query Sequences (domain={DOMAIN}) ===\n")
    print("Router input: (200, 256) per-query states from last DETR decoder layer")
    print("(Not mean-pooled — attention router learns query weighting)\n")

    meta_path = os.path.join(OUT_DIR, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    samples = meta["samples"]
    N = len(samples)
    print(f"Samples in meta.json: {N}")

    # Build lookup from image_id → {path, prompt}
    data_root = DATA_ROOTS.get(DOMAIN)
    if data_root is None:
        raise ValueError(f"Unknown domain '{DOMAIN}'. Available: {list(DATA_ROOTS)}")
    domain_name = DOMAIN if DOMAIN != "wiki-sports" else "wiki-sports_equipment"
    json_path = os.path.join(data_root,
                             f"saco_gold_{domain_name}_test_1.json")
    with open(json_path) as f:
        data = json.load(f)
    id_to_info = {img["id"]: {"path": os.path.join(IMAGE_ROOT, img["file_name"]),
                               "prompt": img["text_input"]}
                  for img in data["images"]}

    wrapper   = SAM3Wrapper()
    all_embs  = []
    skipped   = 0

    for s in tqdm(samples, desc="Extracting full DETR embs"):
        info = id_to_info.get(s["image_id"])
        if info is None or not os.path.exists(info["path"]):
            all_embs.append(np.zeros((200, 256), dtype=np.float32))
            skipped += 1
            continue
        try:
            img = Image.open(info["path"]).convert("RGB")
            pv, ids = wrapper.preprocess(img, info["prompt"])
            hs, backbone_lhs, text_emb_detr, _, _ = wrapper.extract(pv, ids)
            emb_full = wrapper.extract_detr_emb_full(
                hs, backbone_lhs, text_emb_detr, pv, layer_idx=32)
            all_embs.append(emb_full.cpu().numpy())       # (200, 256)
        except Exception as e:
            all_embs.append(np.zeros((200, 256), dtype=np.float32))
            skipped += 1

    arr = np.array(all_embs, dtype=np.float32)            # (N, 200, 256)
    out_path = os.path.join(OUT_DIR, "detr_embs_full.npy")
    np.save(out_path, arr)

    print(f"\nDone.  Extracted: {N - skipped}  skipped/zeros: {skipped}")
    print(f"Saved: {out_path}  shape={arr.shape}")
    mb = arr.nbytes / 1024**2
    print(f"File size: {mb:.1f} MB")


if __name__ == "__main__":
    main()
