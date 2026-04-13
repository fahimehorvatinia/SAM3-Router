"""
Step 3 of 3: Evaluate the trained CAPR router vs baselines.

Compares four methods on a held-out set of SA-Co metaclip positives:
  1. L32 default     — always use SAM3's final layer (block 31)
  2. Oracle best     — always use the best layer per image (upper bound)
  3. Router hard     — use argmax of router weights (top-1 layer selection)
  4. Router MoE      — weighted blend of all 16 layers (soft routing)

Reports: mean cgF1, mean IoU, pmF1 per method.
Saves:   results/router_eval_results.png

Usage:
    cd capr_clean
    python experiments/eval_router.py
"""
import os, sys, json, random
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam3_wrapper import SAM3Wrapper
from capr_router import load_router
from metrics import compute_cgf1, compute_iou, merge_gt_masks

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE  = ("/home/grads/f/fahimehorvatinia/Documents/newpaper_2026"
              "/saco_gold_data/metaclip/saco_gold_metaclip_test_1.json")
IMAGE_ROOT = "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/metaclip-images"
DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "router_training_data")
OUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

TRAIN_LAYERS = list(range(17, 33))   # must match collect step
N_EVAL       = 200                   # held-out evaluation samples
TRAIN_SEED   = 42                    # seed used during collection (to exclude those)
EVAL_SEED    = 99                    # different seed → fresh held-out set
THRESHOLD    = 0.5


# ── Dataset — held-out (not used in training) ─────────────────────────────────
def load_eval_samples():
    with open(DATA_FILE) as f:
        data = json.load(f)
    anno_map = defaultdict(list)
    for a in data["annotations"]:
        anno_map[a["image_id"]].append(a)

    # Load meta from training to exclude those image_ids
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        train_ids = {s["image_id"] for s in json.load(f)["samples"]}

    pool = []
    for img in data["images"]:
        if img["id"] in train_ids:
            continue                # exclude training samples
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

    random.seed(EVAL_SEED)
    random.shuffle(pool)
    selected = pool[:N_EVAL]
    print(f"Held-out positives available: {len(pool)}")
    print(f"Evaluating on:                {len(selected)}")
    return selected


def resize_mask(pred, gt):
    if pred.shape == gt.shape:
        return pred
    return np.array(Image.fromarray(pred.astype(np.uint8) * 255).resize(
        (gt.shape[1], gt.shape[0]), Image.NEAREST)).astype(bool)


# ── Evaluation loop ────────────────────────────────────────────────────────────
def evaluate(samples, wrapper, router):
    results = {m: dict(cgf1=[], iou=[]) for m in ("l32", "oracle", "hard", "moe")}

    for sample in tqdm(samples, desc="Evaluating"):
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
            gt  = merge_gt_masks(sample["annotations"], sample["height"], sample["width"])
            if gt.shape != (sample["height"], sample["width"]):
                gt = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize(
                    (sample["width"], sample["height"]), Image.NEAREST)).astype(bool)
        except Exception:
            continue
        if gt.sum() == 0:
            continue

        try:
            pv, ids = wrapper.preprocess(img, sample["prompt"])
            hs, backbone_lhs, text_emb_detr, text_emb_router = wrapper.extract(pv, ids)
        except Exception as e:
            print(f"  skip {sample['image_id']}: {e}"); continue

        # ── Router weights ────────────────────────────────────────────────────
        with torch.no_grad():
            txt = text_emb_router.cpu().unsqueeze(0)
            layer_weights = router.get_layer_weights(txt)   # [(layer_idx, weight), ...]
            hard_layer    = router.hard_pick(txt)

        # ── Per-layer sweep: ONE pass, results used directly (no re-runs) ───────
        layer_cgf1 = {}
        layer_iou  = {}
        for l in TRAIN_LAYERS:
            try:
                out  = wrapper.run(img, hs, backbone_lhs, text_emb_detr, pv, layer_idx=l)
                pred = out["union_mask"]
                if pred.shape != gt.shape:
                    pred = resize_mask(pred, gt)
                layer_cgf1[l] = compute_cgf1(pred, gt)
                layer_iou[l]  = compute_iou(pred, gt)
            except Exception:
                layer_cgf1[l] = 0.0
                layer_iou[l]  = 0.0

        # Skip samples where oracle can find nothing (all layers produce zeros).
        if max(layer_cgf1.values()) < 0.01:
            continue

        oracle_layer = max(layer_cgf1, key=layer_cgf1.get)

        # ── Method 1: L32 default — use sweep result directly ─────────────────
        results["l32"]["cgf1"].append(layer_cgf1[32])
        results["l32"]["iou"].append(layer_iou[32])

        # ── Method 2: Oracle — use sweep result for best layer ────────────────
        results["oracle"]["cgf1"].append(layer_cgf1[oracle_layer])
        results["oracle"]["iou"].append(layer_iou[oracle_layer])

        # ── Method 3: Router hard (top-1) — use sweep result for hard_layer ───
        # hard_layer must be in TRAIN_LAYERS; fall back to L32 if not
        rl = hard_layer if hard_layer in layer_cgf1 else 32
        results["hard"]["cgf1"].append(layer_cgf1[rl])
        results["hard"]["iou"].append(layer_iou[rl])

        # ── Method 4: Router MoE — ONE extra call (blend of all layers) ───────
        try:
            out_m = wrapper.run_moe(img, hs, backbone_lhs, text_emb_detr, pv,
                                    layer_weights)
            pred_m = out_m["union_mask"]
            if pred_m.shape != gt.shape:
                pred_m = resize_mask(pred_m, gt)
            results["moe"]["cgf1"].append(compute_cgf1(pred_m, gt))
            results["moe"]["iou"].append(compute_iou(pred_m, gt))
        except Exception:
            results["moe"]["cgf1"].append(0.0)
            results["moe"]["iou"].append(0.0)

    return results


# ── Summary and plot ───────────────────────────────────────────────────────────
def summarise(results, n_eval):
    labels = {"l32": "L32 default", "oracle": "Oracle best",
              "hard": "Router hard", "moe": "Router MoE"}
    order  = ["l32", "oracle", "hard", "moe"]
    colors = ["#d62728", "#ff7f0e", "#2196F3", "#9C27B0"]

    print(f"\n{'Method':>18}  {'cgF1':>8}  {'IoU':>8}  {'pmF1':>8}  {'Δ cgF1 vs L32':>14}")
    print("-" * 65)
    l32_cgf1 = float(np.mean(results["l32"]["cgf1"]))
    for m in order:
        cgf1 = float(np.mean(results[m]["cgf1"]))
        iou  = float(np.mean(results[m]["iou"]))
        pmf1 = float(np.mean([v >= 0.5 for v in results[m]["iou"]]))
        delta = cgf1 - l32_cgf1
        sign  = "+" if delta >= 0 else ""
        print(f"  {labels[m]:>16s}  {cgf1:>8.4f}  {iou:>8.4f}  {pmf1:>8.4f}  {sign}{delta:>+.4f}")

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"CAPR Router Evaluation  (n={n_eval} held-out SA-Co positives)",
                 fontsize=13, fontweight="bold")
    for ax, metric, title in [
        (axes[0], "cgf1", "cgF1 (Dice vs GT)"),
        (axes[1], "iou",  "IoU vs GT"),
        (axes[2], None,   "pmF1 (IoU ≥ 0.5 hit rate)"),
    ]:
        if metric:
            vals = [float(np.mean(results[m][metric])) if results[m][metric] else 0.0
                    for m in order]
        else:
            vals = [float(np.mean([v >= 0.5 for v in results[m]["iou"]])) for m in order]
        bars = ax.bar([labels[m] for m in order], vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, min(1.05, max(vals) * 1.25))
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "router_eval_results.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nPlot saved: {out}")


def quick_diagnostic(samples, wrapper, n=10):
    """Check raw query scores on a few samples to understand the zero issue."""
    print("\n--- Quick diagnostic: raw L32 query scores on first 10 eval samples ---")
    for sample in samples[:n]:
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
            gt  = merge_gt_masks(sample["annotations"], sample["height"], sample["width"])
            pv, ids = wrapper.preprocess(img, sample["prompt"])
            hs, backbone_lhs, text_emb_detr, _ = wrapper.extract(pv, ids)
            out = wrapper.run(img, hs, backbone_lhs, text_emb_detr, pv, layer_idx=32)
            print(f"  id={sample['image_id']:>12d}  q={out['query_score']:.4f}"
                  f"  n_masks={out['n_masks']}  gt_px={gt.sum()}"
                  f"  '{sample['prompt'][:40]}'")
        except Exception as e:
            print(f"  id={sample['image_id']}  ERROR: {e}")
    print()


def main():
    print("\n=== CAPR Router Evaluation ===\n")
    samples = load_eval_samples()
    wrapper = SAM3Wrapper()
    router  = load_router()
    router.eval()

    quick_diagnostic(samples, wrapper)

    results = evaluate(samples, wrapper, router)

    n = len(results["l32"]["cgf1"])
    n_nonzero = sum(v > 0 for v in results["l32"]["cgf1"])
    print(f"\nCompleted: {n} samples  |  non-zero cgF1 at L32: {n_nonzero}/{n}")
    summarise(results, n)


if __name__ == "__main__":
    main()
