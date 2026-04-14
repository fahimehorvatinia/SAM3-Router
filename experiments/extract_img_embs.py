"""
Fast extraction of image embeddings for ALREADY-COLLECTED oracle samples.

Reads meta.json to get the 1368 existing samples, runs ONLY the SAM3 backbone
(no FPN / DETR / layer sweep), and saves img_embs.npy alongside text_embs.npy.

Why this is needed:
    The original collect_oracle_layers.py only saved text_embs.npy.
    The updated CAPR router uses concat(text_emb, img_emb) as input (2048-dim)
    so the router has BOTH concept information AND image-level visual context.

Runtime: ~7-12 min for 1368 samples (backbone only, no 16-layer sweep).

Usage:
    cd capr_clean
    python experiments/extract_img_embs.py
"""
import os, sys, json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_FILE  = ("/home/grads/f/fahimehorvatinia/Documents/newpaper_2026"
              "/saco_gold_data/metaclip/saco_gold_metaclip_test_1.json")
IMAGE_ROOT = "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/metaclip-images"
OUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "router_training_data")


def main():
    print("\n=== CAPR — Extracting Image Embeddings for Existing Oracle Samples ===\n")

    # ── Load existing meta to get sample order ────────────────────────────────
    meta_path = os.path.join(OUT_DIR, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    samples = meta["samples"]   # [{image_id, prompt}, ...]
    N = len(samples)
    print(f"Samples in meta.json: {N}")

    # ── Build image_id → file_name + prompt map from dataset ─────────────────
    with open(DATA_FILE) as f:
        data = json.load(f)
    id_to_info = {}
    for img in data["images"]:
        id_to_info[img["id"]] = {
            "path":   os.path.join(IMAGE_ROOT, img["file_name"]),
            "prompt": img["text_input"],
        }

    # ── Load SAM3 backbone only ───────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM3 backbone on {device}...")

    import os as _os
    _os.environ.setdefault("HF_HOME", _os.path.expanduser("~/.cache/huggingface"))
    from transformers import Sam3Model, Sam3Processor

    SAM3_MODEL_ID = "facebook/sam3"
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, local_files_only=True)
    model     = Sam3Model.from_pretrained(SAM3_MODEL_ID, local_files_only=True).to(device)
    model.eval()
    print(f"SAM3 ready on {device}\n")

    # ── Extract image embeddings ──────────────────────────────────────────────
    img_embs = []
    skipped  = 0

    for s in tqdm(samples, desc="Extracting img embs"):
        iid  = s["image_id"]
        info = id_to_info.get(iid)
        if info is None or not os.path.exists(info["path"]):
            # Keep a zero vector so array indices stay aligned with meta.json
            img_embs.append(np.zeros(1024, dtype=np.float32))
            skipped += 1
            continue

        try:
            img    = Image.open(info["path"]).convert("RGB")
            inputs = processor(images=img, text=info["prompt"], return_tensors="pt")
            pv     = inputs["pixel_values"].to(device)

            with torch.no_grad():
                backbone_out = model.vision_encoder.backbone(pv, output_hidden_states=True)
                # Mean-pool hidden_states[32] over spatial dims (H, W) → (1024,)
                emb = backbone_out.hidden_states[32].mean(dim=(1, 2)).squeeze(0)
                img_embs.append(emb.cpu().float().numpy())
        except Exception as e:
            img_embs.append(np.zeros(1024, dtype=np.float32))
            skipped += 1

    img_embs = np.array(img_embs, dtype=np.float32)
    out_path = os.path.join(OUT_DIR, "img_embs.npy")
    np.save(out_path, img_embs)

    print(f"\nDone.")
    print(f"  Extracted: {N - skipped}  skipped/zeros: {skipped}")
    print(f"  Saved: {out_path}  shape={img_embs.shape}")

    # ── Sanity check alignment with text_embs ────────────────────────────────
    text_embs = np.load(os.path.join(OUT_DIR, "text_embs.npy"))
    assert img_embs.shape == text_embs.shape, \
        f"Shape mismatch: img_embs {img_embs.shape} vs text_embs {text_embs.shape}"
    print(f"  Alignment check passed: img_embs {img_embs.shape} == text_embs {text_embs.shape}")


if __name__ == "__main__":
    main()
