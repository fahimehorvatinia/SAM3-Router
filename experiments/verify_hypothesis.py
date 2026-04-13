"""Hypothesis check: do intermediate backbone layers peak on IL_MCC (presence
discrimination) while the final layer peaks on cgF1/IoU (mask quality)?

Dataset:
  Positives — 35 CONFIRMED SAM3 failures (concept present, L32 found nothing)
  Negatives — 35 randomly sampled true-negatives from subset_gt.json
              (concept absent, no GT mask, image verified to exist)

Fixed query-score threshold = 0.5, identical for every layer.
IL_MCC is computed per-layer across all 70 samples (pos + neg).

Architecture note:
  SAM3 ViT has 32 transformer blocks (numbered 0–31).
  hidden_states has 33 entries (indices 0–32):
    hidden_states[0]   = patch embedding
    hidden_states[k]   = after transformer block k-1  (k=1..32)
    hidden_states[32]  = after transformer block 31  ← SAM3 default (L32)
  layer_idx=32 is therefore the final-block output, NOT out-of-range.

Saves:
  results/verify_hypothesis.csv          — per-layer metrics
  results/verify_hypothesis.png          — IL_MCC / cgF1 / IoU / pmF1 vs layer
  results/verify_hypothesis_case.png     — 4-panel showcase case

Usage:  cd capr_clean && python experiments/verify_hypothesis.py
"""
import os, sys, json, csv, random
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam3_wrapper import SAM3Wrapper
from metrics import compute_cgf1, compute_iou, merge_gt_masks

# ── Config ────────────────────────────────────────────────────────────────────
FAILURES_JSON = (
    "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026"
    "/diagnosis_results/failures_double_check/verification_report.json"
)
GT_JSON = (
    "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026"
    "/diagnosis_results/subset_gt.json"
)
IMAGE_ROOT = "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/metaclip-images"
OUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# layer_idx 17–32:  hidden_states[17..32]
#   hidden_states[17] = after transformer block 16
#   hidden_states[32] = after transformer block 31  ← SAM3 default
LAYERS    = list(range(17, 33))
THRESHOLD = 0.5       # fixed query-score threshold, same for every layer
N_NEG     = 35        # match to number of confirmed positives
SEED      = 42

# overlay colours
GT_COLOR   = np.array([0.10, 0.90, 0.30])
BEST_COLOR = np.array([0.10, 0.55, 1.00])
L32_COLOR  = np.array([1.00, 0.20, 0.20])
ALPHA      = 0.45


# ── Dataset ───────────────────────────────────────────────────────────────────
def load_samples():
    # ── Positives: confirmed SAM3 failures ──────────────────────────────────
    with open(FAILURES_JSON) as f:
        failures = json.load(f)
    confirmed = [e for e in failures
                 if e.get("confirmed_failure") and e.get("verdict") == "CONFIRMED"]

    with open(GT_JSON) as f:
        gt_data = json.load(f)
    img_meta = {i["id"]: i for i in gt_data["images"]}
    anno_map = defaultdict(list)
    for a in gt_data["annotations"]:
        anno_map[a["image_id"]].append(a)

    positives = []
    for e in confirmed:
        iid  = e["image_id"]
        meta = img_meta.get(iid)
        if meta is None:
            continue
        annos   = anno_map.get(iid, [])
        gt_mask = merge_gt_masks(annos, meta["height"], meta["width"])
        positives.append(dict(
            image_id=iid, image_path=e["image_path"], prompt=e["phrase"],
            gt_mask=gt_mask, is_present=True,
            height=meta["height"], width=meta["width"],
        ))

    # ── Negatives: images with no GT annotation in subset_gt.json ───────────
    neg_ids  = {e["image_id"] for e in confirmed}   # don't reuse positives
    neg_pool = []
    for img in gt_data["images"]:
        if anno_map.get(img["id"]) or img["id"] in neg_ids:
            continue
        path = os.path.join(IMAGE_ROOT, img["file_name"])
        if not os.path.exists(path):
            continue
        neg_pool.append(dict(
            image_id=img["id"], image_path=path, prompt=img["text_input"],
            gt_mask=None, is_present=False,
            height=img["height"], width=img["width"],
        ))

    random.seed(SEED)
    negatives = random.sample(neg_pool, min(N_NEG, len(neg_pool)))

    print(f"Positives (confirmed failures): {len(positives)}")
    print(f"Negatives (true-negatives):     {len(negatives)}")
    return positives + negatives


# ── Helpers ───────────────────────────────────────────────────────────────────
def resize_mask(pred, shape):
    if pred.shape == shape:
        return pred
    return np.array(Image.fromarray(pred.astype(np.uint8) * 255).resize(
        (shape[1], shape[0]), Image.NEAREST)).astype(bool)


def get_pred(out, gt_shape):
    """Apply fixed threshold; all-zeros if query_score < THRESHOLD."""
    if out["query_score"] >= THRESHOLD:
        return resize_mask(out["best_mask"], gt_shape)
    return np.zeros(gt_shape, dtype=bool)


# ── Sweep ─────────────────────────────────────────────────────────────────────
def run_sweep(samples, wrapper):
    # One-pass: backbone runs once per sample, FPN+decoder once per layer
    buckets = {l: dict(y_true=[], query_score=[], cgf1=[], iou=[]) for l in LAYERS}

    for sample in tqdm(samples, desc="Samples"):
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
            pv, ids = wrapper.preprocess(img, sample["prompt"])
            hs, backbone_lhs, text_emb, _ = wrapper.extract(pv, ids)
        except Exception as e:
            print(f"  skip {sample['image_id']}: {e}"); continue

        gt       = sample["gt_mask"]
        gt_shape = (sample["height"], sample["width"])
        is_pos   = int(sample["is_present"])

        for l in LAYERS:
            try:
                out = wrapper.run(img, hs, backbone_lhs, text_emb, pv, layer_idx=l)
            except Exception as e:
                print(f"  L{l} err: {e}"); continue

            b = buckets[l]
            b["y_true"].append(is_pos)
            b["query_score"].append(out["query_score"])

            # mask metrics only for positives
            if is_pos and gt is not None:
                pred = get_pred(out, gt_shape)
                b["cgf1"].append(compute_cgf1(pred, gt))
                b["iou"].append(compute_iou(pred, gt))

    return buckets


# ── Aggregate ─────────────────────────────────────────────────────────────────
def aggregate(buckets):
    rows = {m: [] for m in ("IL_MCC", "cgF1", "IoU", "pmF1")}
    for l in LAYERS:
        b = buckets[l]

        # IL_MCC — detection across pos + neg (main hypothesis metric)
        y_true = b["y_true"]
        y_pred = [1 if q >= THRESHOLD else 0 for q in b["query_score"]]
        if len(set(y_true)) >= 2:
            il_mcc = float(matthews_corrcoef(y_true, y_pred))
        else:
            il_mcc = 0.0
        rows["IL_MCC"].append(il_mcc)

        # mask quality — positives only
        rows["cgF1"].append(float(np.mean(b["cgf1"])) if b["cgf1"] else 0.0)
        rows["IoU"].append(float(np.mean(b["iou"]))   if b["iou"]  else 0.0)
        rows["pmF1"].append(float(np.mean([iou >= 0.5 for iou in b["iou"]])) if b["iou"] else 0.0)

    csv_path = os.path.join(OUT_DIR, "verify_hypothesis.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "IL_MCC", "cgF1", "IoU", "pmF1"])
        for i, l in enumerate(LAYERS):
            w.writerow([l] + [f"{rows[m][i]:.4f}" for m in ("IL_MCC", "cgF1", "IoU", "pmF1")])
    print(f"CSV saved: {csv_path}")
    return rows


# ── Line chart ────────────────────────────────────────────────────────────────
def save_plot(rows, n_pos, n_neg):
    fig, ax = plt.subplots(figsize=(11, 5))
    styles = [
        ("IL_MCC", "#2196F3", "o", "--", 2.5),
        ("cgF1",   "#FF9800", "s", "-",  2.0),
        ("IoU",    "#4CAF50", "^", "-",  2.0),
        ("pmF1",   "#9C27B0", "D", "-.", 2.0),
    ]
    for metric, color, marker, ls, lw in styles:
        ax.plot(LAYERS, rows[metric], color=color, marker=marker,
                linestyle=ls, linewidth=lw, markersize=6, label=metric)

    # block-number x-axis labels (hidden_state index → block index)
    ax.axvline(x=32, color="red", linestyle=":", linewidth=1.5, label="L32: SAM3 default (block 31)")
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{l}\n(blk {l-1})" for l in LAYERS], fontsize=8)
    ax.set_xlabel("Backbone layer  (hidden_state index / transformer block)", fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Layer sweep — {n_pos} confirmed failures (pos) + {n_neg} true-negatives (neg)\n"
        f"Fixed query threshold = {THRESHOLD}  |  "
        "Hypothesis: IL_MCC peaks earlier than cgF1/IoU",
        fontsize=11, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.15, 1.05)
    out = os.path.join(OUT_DIR, "verify_hypothesis.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Plot saved: {out}")


# ── 4-panel case viz ──────────────────────────────────────────────────────────

# Palette for individual instances (up to 8 colours)
INSTANCE_COLORS = [
    np.array([0.10, 0.55, 1.00]),   # blue
    np.array([1.00, 0.40, 0.10]),   # orange
    np.array([0.80, 0.10, 0.90]),   # magenta
    np.array([0.10, 0.85, 0.85]),   # cyan
    np.array([1.00, 0.85, 0.10]),   # yellow
    np.array([0.10, 0.85, 0.35]),   # green
    np.array([0.95, 0.15, 0.30]),   # red
    np.array([0.60, 0.40, 1.00]),   # violet
]


def overlay_single(img_arr, mask, color):
    """Single-color overlay (used for GT)."""
    out = img_arr.copy()
    out[mask] = (1 - ALPHA) * img_arr[mask] + ALPHA * color
    return np.clip(out, 0, 1)


def overlay_instances(img_arr, masks, gt_shape):
    """Overlay each detected instance with a distinct colour.
    masks: list of bool arrays — ALL instances returned by the wrapper.
    Returns the composited image and a legend list of (color, label) pairs.
    """
    out = img_arr.copy()
    legend = []
    for i, m in enumerate(masks):
        m_rs = resize_mask(m, gt_shape)
        color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
        out[m_rs] = (1 - ALPHA) * img_arr[m_rs] + ALPHA * color
        legend.append((color, f"inst {i+1}"))
    return np.clip(out, 0, 1), legend


def get_union(out, gt_shape):
    """Union of all detected instances resized to gt_shape.
    Falls back to all-zeros if query_score < THRESHOLD."""
    if out["query_score"] < THRESHOLD:
        return np.zeros(gt_shape, dtype=bool), []
    masks = out["masks"]
    if not masks:
        masks = [out["best_mask"]]
    union = np.zeros(gt_shape, dtype=bool)
    for m in masks:
        union |= resize_mask(m, gt_shape)
    return union, masks


def make_case_viz(sample, wrapper, best_int_l, rows):
    img = Image.open(sample["image_path"]).convert("RGB")
    pv, ids = wrapper.preprocess(img, sample["prompt"])
    hs, backbone_lhs, text_emb, _ = wrapper.extract(pv, ids)
    out_best = wrapper.run(img, hs, backbone_lhs, text_emb, pv, layer_idx=best_int_l)
    out_l32  = wrapper.run(img, hs, backbone_lhs, text_emb, pv, layer_idx=32)

    gt       = sample["gt_mask"]
    gt_shape = gt.shape

    # Union of all instances for metrics; individual masks for colour viz
    union_best, masks_best = get_union(out_best, gt_shape)
    union_l32,  masks_l32  = get_union(out_l32,  gt_shape)

    def m(out, union):
        cgf1 = compute_cgf1(union, gt)
        iou  = compute_iou(union, gt)
        n    = len(out["masks"]) if out["query_score"] >= THRESHOLD else 0
        return dict(presence=out["presence"], query=out["query_score"],
                    detected="Yes" if out["query_score"] >= THRESHOLD else "No",
                    n_inst=n, cgf1=cgf1, iou=iou, pmf1=1.0 if iou >= 0.5 else 0.0)

    mb,  ml32  = m(out_best, union_best), m(out_l32, union_l32)
    il_best    = rows["IL_MCC"][LAYERS.index(best_int_l)]
    il_l32     = rows["IL_MCC"][LAYERS.index(32)]

    img_arr = np.array(img).astype(float) / 255.0

    # Build panel images: multi-colour instance overlays
    arr_best, leg_best = overlay_instances(img_arr, masks_best, gt_shape)
    arr_l32,  leg_l32  = overlay_instances(img_arr, masks_l32,  gt_shape)

    panels = [
        (img_arr,                             "Original image",
            None,  None,  []),
        (overlay_single(img_arr, gt, GT_COLOR),
            f"GT mask  (green)  [{gt.sum()} px]",
            None,  None,  [(GT_COLOR, "GT")]),
        (arr_best,
            f"Best intermediate: L{best_int_l}  block {best_int_l-1}",
            mb,    il_best, leg_best),
        (arr_l32,
            "SAM3 default: L32  block 31",
            ml32,  il_l32,  leg_l32),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(
        f'Showcase: "{sample["prompt"]}"  (image_id={sample["image_id"]})\n'
        f'Fixed threshold = {THRESHOLD}  |  '
        f'Each colour = one detected instance  |  '
        f'Metrics computed on UNION of all instances vs GT',
        fontsize=11, fontweight="bold")

    def fmt(mv, il):
        return (f"detected   = {mv['detected']}\n"
                f"instances  = {mv['n_inst']}\n"
                f"presence   = {mv['presence']:.3f}\n"
                f"query_score= {mv['query']:.3f}\n"
                f"cgF1(union)= {mv['cgf1']:.3f}\n"
                f"IoU (union)= {mv['iou']:.3f}\n"
                f"pmF1       = {mv['pmf1']:.1f}\n"
                f"IL_MCC(agg)= {il:.3f}")

    for ax, (arr, title, mv, il, legend) in zip(axes, panels):
        ax.imshow(arr)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
        if mv is not None:
            ax.text(0.02, 0.02, fmt(mv, il), transform=ax.transAxes, fontsize=8.5,
                    verticalalignment="bottom", color="white",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.65))
        # colour legend for detected instances
        if legend:
            from matplotlib.patches import Patch
            handles = [Patch(facecolor=c, label=lbl) for c, lbl in legend]
            ax.legend(handles=handles, loc="upper right", fontsize=7,
                      framealpha=0.7, facecolor="black", labelcolor="white",
                      edgecolor="none", handlelength=1.0, borderpad=0.4)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "verify_hypothesis_case.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Case viz saved: {out_path}")
    print(f"  L{best_int_l}: detected={mb['detected']}  cgF1={mb['cgf1']:.3f}  "
          f"IoU={mb['iou']:.3f}  IL_MCC(agg)={il_best:.3f}")
    print(f"  L32:          detected={ml32['detected']}  cgF1={ml32['cgf1']:.3f}  "
          f"IoU={ml32['iou']:.3f}  IL_MCC(agg)={il_l32:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Hypothesis Verification ===")
    print(f"SAM3 architecture: 32 transformer blocks (0–31), 33 hidden states (indices 0–32)")
    print(f"layer_idx=32 → hidden_states[32] = after block 31 = SAM3 default (CORRECT)\n")

    samples   = load_samples()
    positives = [s for s in samples if s["is_present"]]
    negatives = [s for s in samples if not s["is_present"]]
    wrapper   = SAM3Wrapper()

    # Sanity-check hidden state count on first sample
    try:
        img0 = Image.open(samples[0]["image_path"]).convert("RGB")
        pv0, ids0 = wrapper.preprocess(img0, samples[0]["prompt"])
        hs0, _, _, _ = wrapper.extract(pv0, ids0)
        print(f"Hidden states: {len(hs0)} entries  (expected 33)")
        print(f"  hs[0]  shape: {hs0[0].shape}   ← patch embedding")
        print(f"  hs[32] shape: {hs0[32].shape}  ← after block 31 (SAM3 default)\n")
        del img0, pv0, ids0, hs0
    except Exception as e:
        print(f"  sanity check failed: {e}\n")

    # ── Pre-filter: keep only positives where the WRAPPER also fails at L32 ──
    # The confirmed failures were identified using the native SAM3 repo
    # (double_check_failures.py, build_sam3_image_model). Our wrapper uses
    # HuggingFace transformers — a different implementation with different
    # thresholds and scoring. A case "confirmed failed" in native SAM3 may
    # still produce a mask in the HF wrapper. We must filter to cases where
    # the wrapper ITSELF produces no mask at L32, so the experiment is
    # internally consistent (same codebase defines both baseline and sweep).
    print("\nPre-filtering: checking wrapper behaviour at L32 for each positive sample...")
    hf_failures = []
    for sample in tqdm(positives, desc="L32 pre-filter"):
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
            pv, ids = wrapper.preprocess(img, sample["prompt"])
            hs, backbone_lhs, text_emb, _ = wrapper.extract(pv, ids)
            out = wrapper.run(img, hs, backbone_lhs, text_emb, pv, layer_idx=32)
            if out["query_score"] < THRESHOLD:
                hf_failures.append(sample)
        except Exception as e:
            print(f"  skip {sample['image_id']}: {e}")

    print(f"Positives where wrapper ALSO fails at L32 (q < {THRESHOLD}): "
          f"{len(hf_failures)} / {len(positives)}")
    if len(hf_failures) == 0:
        print("WARNING: No positives fail in the HF wrapper at L32. "
              "The confirmed failures are native-SAM3-specific. "
              "Running sweep on original 35 positives anyway, but treat L32 as upper bound.")
        hf_failures = positives

    samples = hf_failures + negatives
    print(f"Final dataset: {len(hf_failures)} pos + {len(negatives)} neg = {len(samples)} samples\n")

    buckets = run_sweep(samples, wrapper)
    rows    = aggregate(buckets)

    print(f"\n{'Layer':>6}  {'blk':>4}  {'IL_MCC':>8}  {'cgF1':>8}  {'IoU':>8}  {'pmF1':>8}")
    print("-" * 58)
    for i, l in enumerate(LAYERS):
        tag = "  ← SAM3 default" if l == 32 else ""
        print(f"  {l:4d}  {l-1:4d}  {rows['IL_MCC'][i]:>+8.4f}  {rows['cgF1'][i]:>8.4f}"
              f"  {rows['IoU'][i]:>8.4f}  {rows['pmF1'][i]:>8.4f}{tag}")

    best = {m: LAYERS[int(np.argmax(rows[m]))] for m in ("IL_MCC", "cgF1", "IoU", "pmF1")}
    print(f"\nBest IL_MCC @ L{best['IL_MCC']} (block {best['IL_MCC']-1})")
    print(f"Best cgF1   @ L{best['cgF1']}  (block {best['cgF1']-1})")
    if best["IL_MCC"] < best["cgF1"]:
        print("→ HYPOTHESIS CONFIRMED: IL_MCC peaks earlier than cgF1/IoU")
    elif best["IL_MCC"] == best["cgF1"]:
        print("→ HYPOTHESIS PARTIAL: peaks at same layer")
    else:
        print("→ HYPOTHESIS NOT CONFIRMED: IL_MCC peaks later than cgF1/IoU")

    # Best intermediate layer (exclude L32) for showcase
    int_cgf1   = rows["cgF1"][:-1]
    best_int_l = LAYERS[:-1][int(np.argmax(int_cgf1))]
    print(f"\nBest intermediate layer: L{best_int_l} (block {best_int_l-1})  "
          f"aggregate cgF1={max(int_cgf1):.4f}")

    # Showcase: pick the positive where best_int_l gives the highest cgF1
    print(f"\nSelecting best showcase case from {len(hf_failures)} positives @ L{best_int_l}...")
    best_showcase, best_showcase_cgf1 = hf_failures[0], -1.0
    for candidate in hf_failures:
        try:
            img_c = Image.open(candidate["image_path"]).convert("RGB")
            pv_c, ids_c = wrapper.preprocess(img_c, candidate["prompt"])
            hs_c, lhs_c, te_c, _ = wrapper.extract(pv_c, ids_c)
            out_c = wrapper.run(img_c, hs_c, lhs_c, te_c, pv_c, layer_idx=best_int_l)
            pred_c = get_pred(out_c, candidate["gt_mask"].shape)
            c_cgf1 = compute_cgf1(pred_c, candidate["gt_mask"])
            print(f"  id={candidate['image_id']}  '{candidate['prompt']}'  "
                  f"cgF1@L{best_int_l}={c_cgf1:.3f}")
            if c_cgf1 > best_showcase_cgf1:
                best_showcase_cgf1 = c_cgf1
                best_showcase = candidate
        except Exception as e:
            print(f"  skip {candidate['image_id']}: {e}")
    print(f"→ Selected: id={best_showcase['image_id']}  '{best_showcase['prompt']}'  "
          f"cgF1={best_showcase_cgf1:.3f}\n")

    save_plot(rows, len(hf_failures), len(negatives))
    make_case_viz(best_showcase, wrapper, best_int_l, rows)


if __name__ == "__main__":
    main()
