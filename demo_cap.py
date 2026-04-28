"""
Demo: all backbone layers on the "cap" image — correct block naming, all masks overlaid.

SAM3 ViT has 32 transformer blocks: block 0 … block 31.
hidden_states[0]  = patch embedding (before any block)
hidden_states[k]  = after transformer block k-1   (k=1..32)
hidden_states[32] = after block 31 = SAM3 default

For each layer, ALL predicted masks (query_score > 0.3) are overlaid.
Metrics (cgF1, IoU) use the union of all found masks vs the GT mask.

Saves:
  results/demo_masks.png            — grid of all layers, all masks shown
  results/demo_masks_comparison.png — GT | default | best | MoE | hard
  results/demo_metrics_bars.png     — cgF1 and IoU per layer
"""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
import torch

from sam3_wrapper import SAM3Wrapper, SWEEP_LAYERS, SAM3_DEFAULT, layer_label, MASK_SCORE_THRESHOLD
from capr_router import load_router
from metrics import compute_cgf1, compute_iou, merge_gt_masks

# ── Config ───────────────────────────────────────────────────────────────────
IMAGE_PATH = (
    "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026"
    "/metaclip-images/2/100002/metaclip_2_100002_f8c44cc57b6c83911b07dea6.jpeg"
)
PROMPT   = "the cap"
IMAGE_ID = 2860000021
GT_FILE  = "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/diagnosis_results/subset_gt.json"
OUT_DIR  = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

GT_COLOR   = np.array([0.10, 0.90, 0.30])   # green for GT
ALPHA      = 0.50
MASK_COLORS = [
    np.array([0.15, 0.55, 1.00]),   # blue
    np.array([1.00, 0.40, 0.10]),   # orange
    np.array([0.90, 0.15, 0.90]),   # magenta
    np.array([0.10, 0.90, 0.90]),   # cyan
    np.array([1.00, 0.85, 0.10]),   # yellow
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gt_mask():
    with open(GT_FILE) as f:
        data = json.load(f)
    imgs  = {i["id"]: i for i in data["images"]}
    annos = [a for a in data["annotations"] if a["image_id"] == IMAGE_ID]
    img   = imgs[IMAGE_ID]
    gt    = merge_gt_masks(annos, img["height"], img["width"])
    print(f"GT loaded: {gt.sum()} positive pixels, shape {gt.shape}")
    return gt


def resize_mask(mask, target_shape):
    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.array(pil.resize((target_shape[1], target_shape[0]), Image.NEAREST)).astype(bool)


def overlay_all_masks(img_arr, masks, colors=None):
    """Overlay a list of binary masks on img_arr with different colors."""
    out = img_arr.copy()
    for i, m in enumerate(masks):
        c = (colors or MASK_COLORS)[i % len(MASK_COLORS)]
        out[m] = (1 - ALPHA) * img_arr[m] + ALPHA * c
    return np.clip(out, 0, 1)


def panel_style(idx, best_idx):
    if idx == SAM3_DEFAULT:
        return "#d62728"   # red
    if idx == best_idx:
        return "#ff7f0e"   # orange
    return "#222222"       # dark


# ── Figure 1: all layers grid ─────────────────────────────────────────────────

def save_all_layers_grid(image, results, gt_mask, best_idx, path):
    img_arr    = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    gt_resized = resize_mask(gt_mask, img_arr.shape[:2])

    ncols = 7
    nrows = math.ceil((len(results) + 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    fig.patch.set_facecolor("#111111")
    axes = axes.flatten()

    # Panel 0: original + GT in green
    axes[0].imshow(overlay_all_masks(img_arr, [gt_resized], [np.array([0.1, 0.9, 0.3])]))
    axes[0].axis("off")
    axes[0].set_title("Original + GT", fontsize=8, color="white", pad=3,
                      bbox=dict(boxstyle="round,pad=0.3", fc="#1a6b1a", alpha=0.85, ec="none"))

    for i, r in enumerate(results, start=1):
        ax = axes[i]

        # Only overlay masks that passed the 0.5 threshold (r["masks"]).
        # If none passed, show the plain image — no mask, no fallback.
        if r["masks"]:
            all_masks = [resize_mask(m, img_arr.shape[:2]) for m in r["masks"]]
            blended   = overlay_all_masks(img_arr, all_masks)
        else:
            blended   = img_arr
        ax.imshow(blended)
        ax.axis("off")

        bg  = panel_style(r["idx"], best_idx)
        lbl = layer_label(r["idx"])
        tag = ""
        if r["idx"] == SAM3_DEFAULT:
            tag = "  ← SAM3 default"
        elif r["idx"] == best_idx:
            tag = "  ★ best cgF1"

        n   = r["n_masks"]
        title = f"{lbl}{tag}\n{n} mask{'s' if n!=1 else ''} found\ncgF1={r['cgf1']:.3f}  IoU={r['iou']:.3f}"
        ax.set_title(title, fontsize=7, color="white", pad=3,
                     bbox=dict(boxstyle="round,pad=0.25", fc=bg, alpha=0.85, ec="none"))

    for j in range(len(results) + 1, len(axes)):
        axes[j].axis("off")

    legend_handles = [
        Patch(color="#d62728", label="SAM3 default (block 31)"),
        Patch(color="#ff7f0e", label=f"Best cgF1 ({layer_label(best_idx)})"),
        Patch(color="#1a6b1a", label="GT mask (green, panel 0)"),
    ]
    fig.legend(handles=legend_handles, fontsize=9, loc="lower center", ncol=3,
               facecolor="#222", labelcolor="white", bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f'"{PROMPT}" — All backbone layers (embed + block 0 … block 31)\n'
        f'Masks shown only when query_score ≥ {MASK_SCORE_THRESHOLD} (fixed threshold)',
        fontsize=11, fontweight="bold", color="white", y=1.01,
        bbox=dict(boxstyle="round,pad=0.4", fc="#222", alpha=0.85, ec="none")
    )
    plt.tight_layout(pad=0.5)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 2: comparison panel ────────────────────────────────────────────────

def save_comparison(image, results, gt_mask, best_idx, moe_r, hard_r, path):
    img_arr    = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    gt_resized = resize_mask(gt_mask, img_arr.shape[:2])

    def get(idx):
        for r in results:
            if r["idx"] == idx:
                masks = [resize_mask(m, img_arr.shape[:2]) for m in (r["masks"] or [r["best_mask"]])]
                return masks, r["cgf1"], r["iou"], r["n_masks"]
        return [], 0.0, 0.0, 0

    def_masks, def_cgf1, def_iou, def_n = get(SAM3_DEFAULT)
    best_masks, best_cgf1, best_iou, best_n = get(best_idx)
    moe_mask  = resize_mask(moe_r["union_mask"], img_arr.shape[:2])
    hard_mask = resize_mask(hard_r["union_mask"], img_arr.shape[:2])

    panels = [
        (overlay_all_masks(img_arr, [gt_resized], [np.array([0.1, 0.9, 0.3])]),
         "Original + GT mask", "#1a6b1a"),
        (overlay_all_masks(img_arr, def_masks) if def_masks else img_arr,
         f"SAM3 default (block 31)\n{def_n} masks  cgF1={def_cgf1:.3f}  IoU={def_iou:.3f}", "#d62728"),
        (overlay_all_masks(img_arr, best_masks) if best_masks else img_arr,
         f"Oracle best ({layer_label(best_idx)})\n{best_n} masks  cgF1={best_cgf1:.3f}  IoU={best_iou:.3f}", "#ff7f0e"),
        (overlay_all_masks(img_arr, [moe_mask]) if moe_mask.any() else img_arr,
         f"ALSR MoE soft blend\ncgF1={moe_r['cgf1']:.3f}  IoU={moe_r['iou']:.3f}", "#9c27b0"),
        (overlay_all_masks(img_arr, [hard_mask]) if hard_mask.any() else img_arr,
         f"ALSR hard ({layer_label(hard_r['layer'])})\ncgF1={hard_r['cgf1']:.3f}  IoU={hard_r['iou']:.3f}", "#2196f3"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(27, 5))
    fig.patch.set_facecolor("#111111")
    for ax, (img, title, bg) in zip(axes, panels):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=9, color="white", pad=5,
                     bbox=dict(boxstyle="round,pad=0.3", fc=bg, alpha=0.75, ec="none"))

    fig.suptitle(f'"{PROMPT}" — GT | SAM3 default | Oracle best | ALSR MoE | ALSR hard',
                 fontsize=12, fontweight="bold", color="white", y=1.02)
    plt.tight_layout(pad=0.5)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 3: metric bars ─────────────────────────────────────────────────────

def save_metrics_bars(results, best_idx, path):
    labels = [layer_label(r["idx"]) for r in results]
    cgf1s  = [r["cgf1"]   for r in results]
    ious   = [r["iou"]    for r in results]
    counts = [r["n_masks"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(20, 4))
    fig.suptitle(f'"{PROMPT}" — Per-layer metrics vs GT mask  '
                 f'(SAM3 has blocks 0–31; embed = patch embedding before block 0)',
                 fontsize=11, fontweight="bold")

    for ax, vals, title, ycolor in [
        (axes[0], cgf1s,  "cgF1 (Dice vs GT)", "#FF9800"),
        (axes[1], ious,   "IoU vs GT",          "#2196F3"),
        (axes[2], counts, f"# masks found\n(query_score ≥ {MASK_SCORE_THRESHOLD})", "#4CAF50"),
    ]:
        colors = []
        for r in results:
            if r["idx"] == SAM3_DEFAULT:     colors.append("#d62728")
            elif r["idx"] == best_idx:       colors.append("#ff7f0e")
            else:                            colors.append(ycolor)

        x = range(len(vals))
        ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Backbone layer", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        if vals:
            best_i = int(np.argmax(vals))
            ax.annotate(f"best: {labels[best_i]}\n{vals[best_i]:.3f}",
                        xy=(best_i, vals[best_i]),
                        xytext=(best_i, vals[best_i] + max(vals) * 0.08),
                        fontsize=7, ha="center",
                        arrowprops=dict(arrowstyle="->", lw=0.8))

    legend_handles = [
        Patch(color="#d62728", label="SAM3 default (block 31)"),
        Patch(color="#ff7f0e", label=f"Best cgF1 ({layer_label(best_idx)})"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\nImage:  {IMAGE_PATH}")
    print(f"Prompt: '{PROMPT}'")
    print(f"SAM3 default = hidden_states[{SAM3_DEFAULT}] = after block {SAM3_DEFAULT-1}\n")

    image   = Image.open(IMAGE_PATH).convert("RGB")
    gt_mask = load_gt_mask()

    wrapper = SAM3Wrapper()
    pv, ids = wrapper.preprocess(image, PROMPT)
    hidden_states, backbone_lhs, text_emb_detr, text_emb_router = wrapper.extract(pv, ids)

    print(f"\n{'Layer':>25}  {'n_masks':>7}  {'cgF1':>7}  {'IoU':>7}")
    print("-" * 55)

    results = []
    for idx in SWEEP_LAYERS:
        out       = wrapper.run(image, hidden_states, backbone_lhs, text_emb_detr, pv, layer_idx=idx)
        union     = out["union_mask"]
        if union.shape != gt_mask.shape:
            union = resize_mask(union, gt_mask.shape)
        cgf1 = compute_cgf1(union, gt_mask)
        iou  = compute_iou(union,  gt_mask)

        lbl  = layer_label(idx)
        mark = "  ← SAM3 default" if idx == SAM3_DEFAULT else ""
        print(f"  {lbl:25s}  {out['n_masks']:>7d}  {cgf1:>7.4f}  {iou:>7.4f}{mark}")

        results.append({
            "idx":       idx,
            "masks":     out["masks"],
            "best_mask": out["best_mask"],
            "union_mask": union,
            "n_masks":   out["n_masks"],
            "cgf1":      cgf1,
            "iou":       iou,
        })

    best_idx = max(results, key=lambda r: r["cgf1"])["idx"]

    # ── CAPR MoE router ──────────────────────────────────────────────────────
    print(f"\n=== CAPR MoE Router ===")
    router = load_router()
    router.eval()
    with torch.no_grad():
        txt = text_emb_router.cpu().unsqueeze(0)
        layer_weights = router.get_layer_weights(txt)
        hard_layer    = router.hard_pick(txt)
        print("Top 5 weights:")
        for l, w in layer_weights[:5]:
            print(f"  {layer_label(l):25s}  weight={w:.4f}")
        print(f"Hard top-1: {layer_label(hard_layer)}")

    moe_out   = wrapper.run_moe(image, hidden_states, backbone_lhs, text_emb_detr, pv, layer_weights)
    moe_union = moe_out["union_mask"]
    if moe_union.shape != gt_mask.shape:
        moe_union = resize_mask(moe_union, gt_mask.shape)
    moe_r = {
        "union_mask": moe_union,
        "cgf1": compute_cgf1(moe_union, gt_mask),
        "iou":  compute_iou(moe_union,  gt_mask),
    }

    hard_out   = wrapper.run(image, hidden_states, backbone_lhs, text_emb_detr, pv, layer_idx=hard_layer)
    hard_union = hard_out["union_mask"]
    if hard_union.shape != gt_mask.shape:
        hard_union = resize_mask(hard_union, gt_mask.shape)
    hard_r = {
        "layer":      hard_layer,
        "union_mask": hard_union,
        "cgf1":       compute_cgf1(hard_union, gt_mask),
        "iou":        compute_iou(hard_union,  gt_mask),
    }

    # Summary
    def_r = next(r for r in results if r["idx"] == SAM3_DEFAULT)
    best_r = next(r for r in results if r["idx"] == best_idx)
    print(f"\n=== Summary ===")
    print(f"  SAM3 default (block 31) : {def_r['n_masks']} masks  cgF1={def_r['cgf1']:.4f}  IoU={def_r['iou']:.4f}")
    print(f"  Oracle best  ({layer_label(best_idx):10s}) : {best_r['n_masks']} masks  cgF1={best_r['cgf1']:.4f}  IoU={best_r['iou']:.4f}")
    print(f"  CAPR MoE soft            : cgF1={moe_r['cgf1']:.4f}  IoU={moe_r['iou']:.4f}")
    print(f"  CAPR hard    ({layer_label(hard_layer):10s}) : cgF1={hard_r['cgf1']:.4f}  IoU={hard_r['iou']:.4f}")

    print(f"\n=== Saving figures ===")
    save_all_layers_grid(image, results, gt_mask, best_idx,
                         os.path.join(OUT_DIR, "demo_masks.png"))
    save_comparison(image, results, gt_mask, best_idx, moe_r, hard_r,
                    os.path.join(OUT_DIR, "demo_masks_comparison.png"))
    save_metrics_bars(results, best_idx,
                      os.path.join(OUT_DIR, "demo_metrics_bars.png"))


if __name__ == "__main__":
    main()
