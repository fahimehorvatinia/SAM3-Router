"""
Step 1 of 3: Collect oracle layer data for router training.

For each positive image in SA-Co metaclip:
  - Run the SAM3 backbone ONCE → text_emb_router (1024-dim)
  - Sweep layers 17–32 → compute cgF1 per layer vs GT mask
  - Record: image_id, text_emb, cgF1_per_layer

Saves to results/router_training_data/:
  text_embs.npy      — [N, 1024]  float32  text embeddings
  cgf1_matrix.npy    — [N, 16]    float32  cgF1 for each of the 16 layers (17..32)
  meta.json          — image_ids, prompts, layer_list

Usage:
    cd capr_clean
    python experiments/collect_oracle_layers.py
"""
import os, sys, json, random
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam3_wrapper import SAM3Wrapper
from metrics import compute_cgf1, compute_iou, merge_gt_masks

# ── Config ────────────────────────────────────────────────────────────────────
# Use test_1 for training data, test_2 for validation, test_3 for final eval.
DATA_FILE  = ("/home/grads/f/fahimehorvatinia/Documents/newpaper_2026"
              "/saco_gold_data/metaclip/saco_gold_metaclip_test_1.json")
IMAGE_ROOT = "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/metaclip-images"
OUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "router_training_data")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_LAYERS = list(range(17, 33))   # 16 active layers: 17,18,...,32
N_COLLECT    = 1500                  # positives from test_1 for oracle labels
SEED         = 42


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_positives():
    with open(DATA_FILE) as f:
        data = json.load(f)
    anno_map = defaultdict(list)
    for a in data["annotations"]:
        anno_map[a["image_id"]].append(a)

    pool = []
    for img in data["images"]:
        annos = anno_map.get(img["id"], [])
        if not annos:
            continue
        path = os.path.join(IMAGE_ROOT, img["file_name"])
        if not os.path.exists(path):
            continue
        pool.append(dict(
            image_id=img["id"], image_path=path, prompt=img["text_input"],
            height=img["height"], width=img["width"], annotations=annos,
        ))

    random.seed(SEED)
    random.shuffle(pool)
    selected = pool[:N_COLLECT]
    print(f"SA-Co metaclip positives available: {len(pool)}")
    print(f"Collecting:                          {len(selected)}")
    return selected


def resize_mask(pred, gt):
    if pred.shape == gt.shape:
        return pred
    return np.array(Image.fromarray(pred.astype(np.uint8) * 255).resize(
        (gt.shape[1], gt.shape[0]), Image.NEAREST)).astype(bool)


# ── Collection loop ───────────────────────────────────────────────────────────
def collect(samples, wrapper):
    text_embs   = []
    cgf1_matrix = []
    meta        = []

    skipped = 0
    for sample in tqdm(samples, desc="Collecting"):
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            skipped += 1; continue

        # GT mask
        try:
            gt = merge_gt_masks(sample["annotations"], sample["height"], sample["width"])
            if gt.shape != (sample["height"], sample["width"]):
                gt = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize(
                    (sample["width"], sample["height"]), Image.NEAREST)).astype(bool)
        except Exception:
            skipped += 1; continue

        if gt.sum() == 0:
            skipped += 1; continue  # degenerate GT

        # Backbone + text encoder (ONCE per sample)
        try:
            pv, ids = wrapper.preprocess(img, sample["prompt"])
            hs, backbone_lhs, text_emb_detr, text_emb_router = wrapper.extract(pv, ids)
        except Exception as e:
            skipped += 1; continue

        # Sweep layers → cgF1 per layer
        row = []
        for l in TRAIN_LAYERS:
            try:
                out  = wrapper.run(img, hs, backbone_lhs, text_emb_detr, pv, layer_idx=l)
                pred = out["union_mask"]
                if pred.shape != gt.shape:
                    pred = resize_mask(pred, gt)
                row.append(compute_cgf1(pred, gt))
            except Exception:
                row.append(0.0)

        # Skip samples where every layer found nothing (all zeros → no signal)
        if max(row) < 0.01:
            skipped += 1; continue

        text_embs.append(text_emb_router.cpu().float().numpy())  # (1024,)
        cgf1_matrix.append(row)                                   # (16,)
        meta.append(dict(image_id=sample["image_id"], prompt=sample["prompt"]))

    print(f"\nCollected: {len(text_embs)}  skipped: {skipped}")
    return np.array(text_embs, dtype=np.float32), np.array(cgf1_matrix, dtype=np.float32), meta


# ── Save ──────────────────────────────────────────────────────────────────────
def save(text_embs, cgf1_matrix, meta):
    N = len(meta)
    # 70 / 20 / 10 split — deterministic, based on index order (already shuffled)
    n_train = int(N * 0.70)
    n_val   = int(N * 0.20)
    # n_test  = N - n_train - n_val  (remaining 10%)
    splits = dict(
        train = list(range(0,              n_train)),
        val   = list(range(n_train,        n_train + n_val)),
        test  = list(range(n_train + n_val, N)),
    )

    np.save(os.path.join(OUT_DIR, "text_embs.npy"),   text_embs)
    np.save(os.path.join(OUT_DIR, "cgf1_matrix.npy"), cgf1_matrix)
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump({"layer_list": TRAIN_LAYERS, "samples": meta,
                   "splits": splits}, f, indent=2)

    print(f"  70/20/10 split: {len(splits['train'])} train  "
          f"{len(splits['val'])} val  {len(splits['test'])} test")

    print(f"\nSaved to {OUT_DIR}/")
    print(f"  text_embs.npy    : {text_embs.shape}")
    print(f"  cgf1_matrix.npy  : {cgf1_matrix.shape}")
    print(f"  meta.json        : {len(meta)} entries")
    print(f"\nPer-layer mean cgF1:")
    for i, l in enumerate(TRAIN_LAYERS):
        vals = cgf1_matrix[:, i]
        print(f"  L{l} (block {l-1}): mean={vals.mean():.4f}  max={vals.max():.4f}")

    best_layer_counts = np.argmax(cgf1_matrix, axis=1)
    print(f"\nOracle best-layer distribution (index → layer):")
    for i, l in enumerate(TRAIN_LAYERS):
        cnt = (best_layer_counts == i).sum()
        pct = 100 * cnt / len(meta)
        bar = "█" * int(pct)
        print(f"  L{l} (blk {l-1:2d}): {cnt:4d} ({pct:4.1f}%)  {bar}")


def main():
    print("\n=== CAPR Router — Data Collection ===\n")
    print(f"Layers to sweep: {TRAIN_LAYERS}")
    print(f"Samples to collect: {N_COLLECT}\n")

    samples = load_positives()
    wrapper = SAM3Wrapper()
    text_embs, cgf1_matrix, meta = collect(samples, wrapper)
    save(text_embs, cgf1_matrix, meta)


if __name__ == "__main__":
    main()
