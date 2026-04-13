"""
Find the top-10 failure cases: images where SAM3 default (L32) completely misses
(query_score < 0.5) but at least one intermediate layer (L0–L31) detects the
concept with high cgF1.

For each of the top-10 cases, saves a demo_masks-style grid showing all 33 layers.

Steps:
  1. Load all available positives from subset_gt.json
  2. Screen every positive at L32 — keep only those where L32 fails (q < 0.5)
  3. For each L32 failure, sweep all 33 layers and compute cgF1 per layer
  4. Rank by best INTERMEDIATE-layer cgF1 (L0–L31 only)
  5. Save the top-10 as demo_masks_failure_{i:02d}_{prompt_slug}.png

Output folder:  results/failure_cases/

Usage:
    cd capr_clean
    python experiments/find_failure_cases.py
"""
import os, sys, json, math, random
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam3_wrapper import (SAM3Wrapper, SWEEP_LAYERS, SAM3_DEFAULT,
                           layer_label, MASK_SCORE_THRESHOLD)
from metrics import compute_cgf1, compute_iou, merge_gt_masks

# ── Config ────────────────────────────────────────────────────────────────────
GT_JSON    = ("/home/grads/f/fahimehorvatinia/Documents/newpaper_2026"
              "/diagnosis_results/subset_gt.json")
IMAGE_ROOT = "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/metaclip-images"
OUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "failure_cases")
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLD  = MASK_SCORE_THRESHOLD   # 0.5 — same threshold everywhere
TOP_N      = 10
SEED       = 42
ALPHA      = 0.50
MASK_COLORS = [
    np.array([0.15, 0.55, 1.00]),
    np.array([1.00, 0.40, 0.10]),
    np.array([0.90, 0.15, 0.90]),
    np.array([0.10, 0.90, 0.90]),
    np.array([1.00, 0.85, 0.10]),
]


# ── Dataset ───────────────────────────────────────────────────────────────────
def load_all_positives():
    with open(GT_JSON) as f:
        gt_data = json.load(f)
    img_meta = {i["id"]: i for i in gt_data["images"]}
    anno_map = defaultdict(list)
    for a in gt_data["annotations"]:
        anno_map[a["image_id"]].append(a)

    positives = []
    for img in gt_data["images"]:
        annos = anno_map.get(img["id"], [])
        if not annos:
            continue
        path = os.path.join(IMAGE_ROOT, img["file_name"])
        if not os.path.exists(path):
            continue
        try:
            gt = merge_gt_masks(annos, img["height"], img["width"])
            if gt.shape != (img["height"], img["width"]):
                gt = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize(
                    (img["width"], img["height"]), Image.NEAREST)).astype(bool)
        except Exception:
            continue
        positives.append(dict(
            image_id=img["id"], image_path=path, prompt=img["text_input"],
            gt_mask=gt, height=img["height"], width=img["width"],
        ))

    random.seed(SEED)
    random.shuffle(positives)
    print(f"Positives available: {len(positives)}")
    return positives


# ── Helpers ───────────────────────────────────────────────────────────────────
def resize_mask(mask, target_shape):
    if mask.shape == target_shape:
        return mask
    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.array(pil.resize((target_shape[1], target_shape[0]),
                                Image.NEAREST)).astype(bool)


def overlay_masks(img_arr, masks):
    out = img_arr.copy()
    for i, m in enumerate(masks):
        c = MASK_COLORS[i % len(MASK_COLORS)]
        out[m] = (1 - ALPHA) * img_arr[m] + ALPHA * c
    return np.clip(out, 0, 1)


def panel_bg(idx, best_idx):
    if idx == SAM3_DEFAULT: return "#d62728"
    if idx == best_idx:     return "#ff7f0e"
    return "#222222"


# ── Full layer sweep for one image ────────────────────────────────────────────
def sweep_one(wrapper, image, gt_mask, pv, hidden_states, backbone_lhs, text_emb):
    img_shape = (image.size[1], image.size[0])
    results = []
    for idx in SWEEP_LAYERS:
        out  = wrapper.run(image, hidden_states, backbone_lhs, text_emb, pv, layer_idx=idx)
        # masks and union already filtered at THRESHOLD (0.5)
        union = out["union_mask"]
        if union.shape != gt_mask.shape:
            union = resize_mask(union, gt_mask.shape)
        cgf1 = compute_cgf1(union, gt_mask)
        iou  = compute_iou(union, gt_mask)
        results.append(dict(
            idx=idx,
            masks=out["masks"],
            best_mask=out["best_mask"],
            union_mask=union,
            n_masks=out["n_masks"],
            query_score=out["query_score"],
            cgf1=cgf1, iou=iou,
        ))
    return results


# ── Grid figure (same style as demo_masks.png) ───────────────────────────────
def save_grid(image, results, gt_mask, sample, rank, path):
    img_arr    = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    gt_resized = resize_mask(gt_mask, img_arr.shape[:2])

    # best intermediate layer (exclude L32)
    int_results = [r for r in results if r["idx"] != SAM3_DEFAULT]
    best_int    = max(int_results, key=lambda r: r["cgf1"])
    best_idx    = best_int["idx"]
    l32_r       = next(r for r in results if r["idx"] == SAM3_DEFAULT)

    ncols = 7
    nrows = math.ceil((len(results) + 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    fig.patch.set_facecolor("#111111")
    axes = axes.flatten()

    # Panel 0: original + GT
    axes[0].imshow(overlay_masks(img_arr, [gt_resized]))
    axes[0].axis("off")
    axes[0].set_title("Original + GT", fontsize=8, color="white", pad=3,
                       bbox=dict(boxstyle="round,pad=0.3", fc="#1a6b1a", alpha=0.85, ec="none"))

    for i, r in enumerate(results, start=1):
        ax = axes[i]
        if r["masks"]:
            blended = overlay_masks(img_arr,
                                    [resize_mask(m, img_arr.shape[:2]) for m in r["masks"]])
        else:
            blended = img_arr
        ax.imshow(blended)
        ax.axis("off")

        bg  = panel_bg(r["idx"], best_idx)
        lbl = layer_label(r["idx"])
        tag = "  ← SAM3 default" if r["idx"] == SAM3_DEFAULT else (
              "  ★ best cgF1"    if r["idx"] == best_idx     else "")
        title = (f"{lbl}{tag}\n{r['n_masks']} mask{'s' if r['n_masks']!=1 else ''} "
                 f"found\ncgF1={r['cgf1']:.3f}  IoU={r['iou']:.3f}")
        ax.set_title(title, fontsize=7, color="white", pad=3,
                     bbox=dict(boxstyle="round,pad=0.25", fc=bg, alpha=0.85, ec="none"))

    for j in range(len(results) + 1, len(axes)):
        axes[j].axis("off")

    fig.legend(handles=[
        Patch(color="#d62728", label=f"SAM3 default (block 31): "
              f"q={l32_r['query_score']:.3f}  cgF1={l32_r['cgf1']:.3f}"),
        Patch(color="#ff7f0e", label=f"Best intermediate ({layer_label(best_idx)}): "
              f"q={best_int['query_score']:.3f}  cgF1={best_int['cgf1']:.3f}"),
        Patch(color="#1a6b1a", label="GT mask"),
    ], fontsize=9, loc="lower center", ncol=3,
       facecolor="#222", labelcolor="white", bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f'Case #{rank}  |  "{sample["prompt"]}"  (id={sample["image_id"]})\n'
        f'SAM3 default FAILS (q={l32_r["query_score"]:.3f} < {THRESHOLD})  |  '
        f'Best intermediate: {layer_label(best_idx)} (q={best_int["query_score"]:.3f}  '
        f'cgF1={best_int["cgf1"]:.3f})',
        fontsize=11, fontweight="bold", color="white", y=1.01,
        bbox=dict(boxstyle="round,pad=0.4", fc="#222", alpha=0.85, ec="none"))

    plt.tight_layout(pad=0.5)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Finding Top-10 L32 Failure Cases ===\n")
    positives = load_all_positives()
    wrapper   = SAM3Wrapper()

    # ── Step 1: screen all positives at L32 ──────────────────────────────────
    print("\nStep 1 — Screening at L32 (fast pass)...")
    l32_failures = []
    for sample in tqdm(positives, desc="L32 screen"):
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
            pv, ids = wrapper.preprocess(img, sample["prompt"])
            hs, backbone_lhs, text_emb, _ = wrapper.extract(pv, ids)
            out = wrapper.run(img, hs, backbone_lhs, text_emb, pv, layer_idx=32)
            if out["query_score"] < THRESHOLD:
                sample["l32_query"] = out["query_score"]
                l32_failures.append(sample)
        except Exception as e:
            print(f"  skip {sample['image_id']}: {e}")

    print(f"\nL32 failures (q < {THRESHOLD}): {len(l32_failures)} / {len(positives)}")

    # ── Step 2: full sweep for each L32 failure ───────────────────────────────
    print("\nStep 2 — Full 33-layer sweep for each L32 failure...")
    candidates = []
    for sample in tqdm(l32_failures, desc="Full sweep"):
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
            pv, ids = wrapper.preprocess(img, sample["prompt"])
            hs, backbone_lhs, text_emb, _ = wrapper.extract(pv, ids)
            results = sweep_one(wrapper, img, sample["gt_mask"],
                                pv, hs, backbone_lhs, text_emb)
            # best intermediate layer (not L32)
            int_r    = [r for r in results if r["idx"] != SAM3_DEFAULT]
            best_int = max(int_r, key=lambda r: r["cgf1"])
            if best_int["cgf1"] > 0:    # at least one intermediate layer found something
                candidates.append(dict(sample=sample, results=results,
                                       best_int_cgf1=best_int["cgf1"],
                                       best_int_idx=best_int["idx"]))
        except Exception as e:
            print(f"  skip {sample['image_id']}: {e}")

    # ── Step 3: rank and take top-10 ─────────────────────────────────────────
    candidates.sort(key=lambda c: c["best_int_cgf1"], reverse=True)
    top10 = candidates[:TOP_N]

    print(f"\nTop {len(top10)} cases (ranked by best intermediate-layer cgF1):")
    print(f"{'Rank':>5}  {'L32 q':>8}  {'Best int layer':>20}  {'Best cgF1':>10}  Prompt")
    print("-" * 80)
    for i, c in enumerate(top10, 1):
        s = c["sample"]
        print(f"  {i:3d}  {s['l32_query']:>8.4f}  "
              f"{layer_label(c['best_int_idx']):>20s}  {c['best_int_cgf1']:>10.4f}  "
              f"'{s['prompt']}'")

    # ── Step 4: save grids ────────────────────────────────────────────────────
    print(f"\nSaving {len(top10)} grid figures to {OUT_DIR}/ ...")
    for rank, c in enumerate(top10, 1):
        sample  = c["sample"]
        results = c["results"]
        img     = Image.open(sample["image_path"]).convert("RGB")
        slug    = sample["prompt"][:35].replace(" ", "_").replace("/", "-")
        fname   = f"failure_{rank:02d}_{slug}.png"
        save_grid(img, results, sample["gt_mask"], sample, rank,
                  os.path.join(OUT_DIR, fname))

    print(f"\nDone. {len(top10)} PNGs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
