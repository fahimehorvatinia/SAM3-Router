"""
Full layer sweep evaluation — IL_MCC, cgF1, pmF1 for every backbone layer.

For each layer in SWEEP_LAYERS:
  - Run SAM3 on N_POS positive + N_NEG negative samples
  - Compute IL_MCC  (concept recognition, needs both pos & neg)
  - Compute cgF1    (mask Dice/F1, positive samples only)
  - Compute pmF1    (IoU >= 0.5 hit rate, positive samples only)

Saves:
  results/eval_raw.csv          — per-sample per-layer results
  results/eval_summary.csv      — per-layer aggregated metrics
  results/eval_metrics_plot.png — IL_MCC / cgF1 / pmF1 vs layer index

Usage:
    cd capr_clean
    python evaluate.py
"""
import os, sys, json, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

from sam3_wrapper import SAM3Wrapper, SWEEP_LAYERS
from metrics import (
    compute_il_mcc, compute_cgf1, compute_iou, compute_pmf1,
    merge_gt_masks,
)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_FILE      = "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/saco_gold_data/metaclip/saco_gold_metaclip_test_1.json"
IMAGE_ROOT     = "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/metaclip-images"
OUT_DIR        = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

N_POS          = 100    # positive samples (concept IS present, has GT mask)
N_NEG          = 100    # negative samples (concept NOT present, no GT mask)
RANDOM_SEED    = 42

# Test every other layer to save time (still covers the full range)
EVAL_LAYERS    = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset():
    print(f"Loading dataset from {DATA_FILE} ...")
    with open(DATA_FILE) as f:
        data = json.load(f)

    anno_map = defaultdict(list)
    for a in data["annotations"]:
        anno_map[a["image_id"]].append(a)

    positives, negatives = [], []
    for img in data["images"]:
        path = os.path.join(IMAGE_ROOT, img["file_name"])
        if not os.path.exists(path):
            continue
        annos = anno_map.get(img["id"], [])
        entry = {
            "pair_id":    img["id"],
            "image_path": path,
            "prompt":     img["text_input"],
            "width":      img["width"],
            "height":     img["height"],
            "is_positive": bool(annos),
            "annotations": annos,
        }
        if entry["is_positive"]:
            positives.append(entry)
        else:
            negatives.append(entry)

    random.seed(RANDOM_SEED)
    pos = random.sample(positives, min(N_POS, len(positives)))
    neg = random.sample(negatives, min(N_NEG, len(negatives)))

    # Decode GT masks for positives
    for e in pos:
        e["gt_mask"] = merge_gt_masks(e["annotations"], e["height"], e["width"])
    for e in neg:
        e["gt_mask"] = None

    print(f"  {len(pos)} positive + {len(neg)} negative samples")
    return pos + neg


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_evaluation():
    samples = load_dataset()
    wrapper = SAM3Wrapper()

    all_rows = []

    for sample in tqdm(samples, desc="Samples"):
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            print(f"  skip {sample['pair_id']}: {e}")
            continue

        # Run backbone ONCE per sample — shared across all layers
        try:
            pv, ids = wrapper.preprocess(image, sample["prompt"])
            hidden_states, backbone_lhs, text_emb_detr, _ = wrapper.extract(pv, ids)
        except Exception as e:
            print(f"  backbone error {sample['pair_id']}: {e}")
            continue

        for layer_idx in EVAL_LAYERS:
            try:
                out = wrapper.run(image, hidden_states, backbone_lhs, text_emb_detr, pv, layer_idx=layer_idx)
            except Exception as e:
                print(f"  layer {layer_idx} error {sample['pair_id']}: {e}")
                continue

            presence    = out["presence"]
            pred_mask   = out["mask"]
            is_positive = sample["is_positive"]
            gt_mask     = sample.get("gt_mask")

            # Segmentation metrics (positive samples only)
            cgf1 = iou = None
            if is_positive and gt_mask is not None:
                # Resize pred_mask to gt_mask shape if needed
                if pred_mask.shape != gt_mask.shape:
                    from PIL import Image as PILImage
                    pred_pil  = PILImage.fromarray(pred_mask.astype(np.uint8) * 255)
                    pred_mask = np.array(
                        pred_pil.resize((gt_mask.shape[1], gt_mask.shape[0]), PILImage.NEAREST)
                    ).astype(bool)
                cgf1 = compute_cgf1(pred_mask, gt_mask)
                iou  = compute_iou(pred_mask, gt_mask)

            all_rows.append({
                "layer_idx":    layer_idx,
                "pair_id":      sample["pair_id"],
                "prompt":       sample["prompt"],
                "is_positive":  int(is_positive),
                "presence":     float(presence),
                "predicted":    int(presence > 0.5),
                "cgf1":         cgf1,
                "iou":          iou,
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT_DIR, "eval_raw.csv"), index=False)
    print(f"\nRaw results saved ({len(df)} rows)")
    return df


# ── Aggregate metrics per layer ───────────────────────────────────────────────

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer_idx in EVAL_LAYERS:
        sub = df[df["layer_idx"] == layer_idx]
        if sub.empty:
            continue

        # IL_MCC — needs all samples (pos + neg)
        il_mcc_metrics = compute_il_mcc(
            sub["is_positive"].tolist(),
            sub["presence"].tolist(),
        )

        # cgF1 — positive samples only
        pos = sub[sub["is_positive"] == 1]
        cgf1_vals = pos["cgf1"].dropna().tolist()
        iou_vals  = pos["iou"].dropna().tolist()
        mean_cgf1 = float(np.mean(cgf1_vals)) if cgf1_vals else 0.0
        pmf1      = compute_pmf1(iou_vals)

        rows.append({
            "layer":    layer_idx,
            "IL_MCC":  il_mcc_metrics["il_mcc"],
            "cgF1":    round(mean_cgf1, 4),
            "pmF1":    round(pmf1, 4),
            "recall":  il_mcc_metrics["recall"],
            "precision": il_mcc_metrics["precision"],
            "tp": il_mcc_metrics["tp"], "tn": il_mcc_metrics["tn"],
            "fp": il_mcc_metrics["fp"], "fn": il_mcc_metrics["fn"],
            "n_pos":   int((sub["is_positive"] == 1).sum()),
            "n_neg":   int((sub["is_positive"] == 0).sum()),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "eval_summary.csv"), index=False)
    return summary


# ── Plot ──────────────────────────────────────────────────────────────────────

def save_metrics_plot(summary: pd.DataFrame):
    layers  = summary["layer"].tolist()
    il_mcc  = summary["IL_MCC"].tolist()
    cgf1    = summary["cgF1"].tolist()
    pmf1    = summary["pmF1"].tolist()

    # Highlight SAM3 default (layer 32)
    default_idx = layers.index(32) if 32 in layers else -1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle("CAPR Layer Sweep — SAM3 Paper Metrics vs Backbone Layer",
                 fontsize=13, fontweight="bold")

    for ax, vals, title, color in [
        (axes[0], il_mcc, "IL_MCC",  "#2196F3"),
        (axes[1], cgf1,   "cgF1",    "#FF9800"),
        (axes[2], pmf1,   "pmF1",    "#4CAF50"),
    ]:
        bar_colors = [
            "#d62728" if l == 32 else color for l in layers
        ]
        ax.bar([str(l) for l in layers], vals, color=bar_colors, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Backbone layer", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

        # Annotate SAM3 default bar
        if default_idx >= 0:
            ax.annotate("SAM3\ndefault",
                        xy=(default_idx, vals[default_idx]),
                        xytext=(default_idx, vals[default_idx] + max(vals) * 0.05),
                        fontsize=7, color="#d62728", ha="center")

        # Annotate best bar
        best_idx = int(np.argmax(vals))
        ax.annotate(f"best: layer {layers[best_idx]}\n{vals[best_idx]:.3f}",
                    xy=(best_idx, vals[best_idx]),
                    xytext=(best_idx, vals[best_idx] + max(vals) * 0.10),
                    fontsize=7, color="black", ha="center",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "eval_metrics_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"Layer Sweep Evaluation — IL_MCC / cgF1 / pmF1")
    print(f"Layers:   {EVAL_LAYERS}")
    print(f"Samples:  {N_POS} positive + {N_NEG} negative")
    print(f"{'='*60}\n")

    df      = run_evaluation()
    summary = compute_summary(df)

    print(f"\n{'='*60}")
    print(f"{'Layer':>7}  {'IL_MCC':>8}  {'cgF1':>8}  {'pmF1':>8}  {'Recall':>8}  {'Prec':>8}")
    print("-" * 60)
    for _, row in summary.iterrows():
        marker = "  ← SAM3 default" if row["layer"] == 32 else ""
        print(f"  {int(row['layer']):4d}    {row['IL_MCC']:>+7.4f}  "
              f"{row['cgF1']:>8.4f}  {row['pmF1']:>8.4f}  "
              f"{row['recall']:>8.4f}  {row['precision']:>8.4f}{marker}")
    print(f"{'='*60}")

    best_il   = summary.loc[summary["IL_MCC"].idxmax()]
    best_cgf1 = summary.loc[summary["cgF1"].idxmax()]
    best_pmf1 = summary.loc[summary["pmF1"].idxmax()]
    print(f"\nBest IL_MCC : layer {int(best_il['layer'])}  = {best_il['IL_MCC']:+.4f}")
    print(f"Best cgF1   : layer {int(best_cgf1['layer'])}  = {best_cgf1['cgF1']:.4f}")
    print(f"Best pmF1   : layer {int(best_pmf1['layer'])}  = {best_pmf1['pmF1']:.4f}")

    save_metrics_plot(summary)
    print(f"\nResults in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
