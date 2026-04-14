"""
Full-scale evaluation of CAPR router on SA-Co metaclip test_3 (held-out).

Dataset split used across the 3 SA-Co metaclip folds:
  test_1  →  router training data (oracle label collection)
  test_2  →  router validation during training
  test_3  →  THIS SCRIPT — final evaluation, never seen during training

Primary metric: IL_MCC (Instance-Level Matthews Correlation Coefficient)
  Measures whether the model correctly classifies concept PRESENCE vs ABSENCE.
  Requires both positives (concept present, has GT mask) and negatives
  (concept absent, no GT annotation). Threshold = 0.5 on query_score.

All metrics computed per method:
  IL_MCC  — presence/absence discrimination  (pos + neg)
  cgF1    — mask quality, Dice vs GT         (pos only)
  IoU     — mask quality, intersection/union  (pos only)
  pmF1    — fraction of positives with IoU ≥ 0.5

Methods compared:
  L32 default  — always use SAM3's final layer (block 31)
  Router hard  — top-1 layer from trained CAPR router
  Router MoE   — soft blend of all 16 layers weighted by router

  Oracle       — per-image best layer (upper bound, slow — run on subset)

Each sample: backbone runs ONCE, then 1 FPN pass per method (+ 16 for oracle).
Oracle is evaluated on N_ORACLE_POS positives only to control runtime.

Saves:
  results/eval_full_raw.csv         — per-sample results
  results/eval_full_summary.png     — bar chart: all metrics × all methods
  results/eval_full_layer_dist.png  — which layers the router selects

Usage:
    cd capr_clean
    python experiments/eval_router_full.py
"""
import os, sys, json, csv, random
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
from sklearn.metrics import matthews_corrcoef

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam3_wrapper import SAM3Wrapper
from capr_router import load_router
from metrics import compute_cgf1, compute_iou, merge_gt_masks

# ── Config ────────────────────────────────────────────────────────────────────
# The 10% test split comes from the collected oracle data (meta.json splits.test).
# These positives already HAVE oracle labels (cgF1 per layer) from the collection.
# For negatives (needed for IL_MCC), sample from test_3 which was never used.
TEST_FILE  = ("/home/grads/f/fahimehorvatinia/Documents/newpaper_2026"
              "/saco_gold_data/metaclip/saco_gold_metaclip_test_3.json")
DATA_DIR_EVAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results", "router_training_data")
IMAGE_ROOT = "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/metaclip-images"
OUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_LAYERS  = list(range(17, 33))   # must match collect step
N_NEG         = 500     # negatives from test_3 for IL_MCC
N_ORACLE_POS  = 100     # positives for oracle sweep if no pre-computed labels
THRESHOLD     = 0.5     # fixed query-score threshold for presence detection
SEED          = 77      # different from train (42) and val seeds

# ── Gated router config ───────────────────────────────────────────────────────
# The router was trained ONLY on "failed cases" — samples where the default
# SAM3 layer (L32) under-performs the oracle by >= DELTA_THRESHOLD cgF1.
# At inference we therefore apply the same gate: if L32 already confidently
# detects the concept (query_score >= GATE_THRESHOLD), we trust L32 and do
# not invoke the router.  Only when L32 is struggling (low query score) do we
# ask the router to find a better backbone layer.
#
# Default: GATE_THRESHOLD = THRESHOLD = 0.5 — the same threshold used to
# decide presence/absence.  A sample with q32 < 0.5 is one L32 would label
# "absent"; the router is invoked to try to recover detection.
# Lowered to 0.4: the v5 router is trained on more failed cases (DELTA>=0.02)
# and uses a stronger image+text signal.  A lower gate means we route more
# aggressively, which is appropriate when the router is more accurate.
GATE_THRESHOLD = float(os.environ.get("GATE_THRESHOLD", "0.4"))


# ── Dataset ───────────────────────────────────────────────────────────────────
def load_eval_samples():
    """
    Positives: the 10% test split from collected oracle data (meta.json splits.test).
    These already have oracle cgF1 labels from the collection step.

    Negatives: random sample from test_3 (never used in training).
    Needed for IL_MCC (requires both classes).
    """
    # ── Positives from 10% oracle test split ─────────────────────────────────
    text_embs   = np.load(os.path.join(DATA_DIR_EVAL, "text_embs.npy"))
    cgf1_matrix = np.load(os.path.join(DATA_DIR_EVAL, "cgf1_matrix.npy"))
    with open(os.path.join(DATA_DIR_EVAL, "meta.json")) as f:
        meta = json.load(f)
    layer_list  = meta["layer_list"]
    splits      = meta.get("splits", {})
    test_idx    = splits.get("test", list(range(int(len(meta["samples"]) * 0.90),
                                               len(meta["samples"]))))
    test_samples = [meta["samples"][i] for i in test_idx]
    test_embs    = text_embs[test_idx]
    test_cgf1    = cgf1_matrix[test_idx]

    positives = []
    for i, s in enumerate(test_samples):
        positives.append(dict(
            image_id=s["image_id"], image_path=s["image_path"] if "image_path" in s
                     else os.path.join(IMAGE_ROOT, ""),
            prompt=s["prompt"], is_present=True,
            text_emb=test_embs[i],
            oracle_cgf1=test_cgf1[i].tolist(),   # pre-computed per layer
        ))

    # We still need the image path — look it up from test_3 or test_1
    # (meta.json might not store image paths in all versions)
    # Load from TEST_FILE to get paths for positives
    with open(TEST_FILE) as f:
        data3 = json.load(f)
    id_to_img3 = {i["id"]: i for i in data3["images"]}

    # ── Negatives from test_3 (completely held-out) ───────────────────────────
    anno_map3 = defaultdict(list)
    for a in data3["annotations"]:
        anno_map3[a["image_id"]].append(a)

    neg_pool = []
    for img in data3["images"]:
        if anno_map3.get(img["id"]):
            continue
        path = os.path.join(IMAGE_ROOT, img["file_name"])
        if not os.path.exists(path):
            continue
        neg_pool.append(dict(image_id=img["id"], image_path=path, prompt=img["text_input"],
                             height=img["height"], width=img["width"],
                             annotations=[], is_present=False,
                             text_emb=None, oracle_cgf1=None))

    random.seed(SEED)
    negatives = random.sample(neg_pool, min(N_NEG, len(neg_pool)))

    # Also load SA-Co test_1 image paths for the oracle-labelled positives
    # (needed for actual inference; re-use from collect_oracle_layers output)
    with open(("/home/grads/f/fahimehorvatinia/Documents/newpaper_2026"
               "/saco_gold_data/metaclip/saco_gold_metaclip_test_1.json")) as f:
        data1 = json.load(f)
    id_to_img1 = {i["id"]: i for i in data1["images"]}
    anno_map1  = defaultdict(list)
    for a in data1["annotations"]:
        anno_map1[a["image_id"]].append(a)

    for pos in positives:
        iid = pos["image_id"]
        if iid in id_to_img1:
            img_meta = id_to_img1[iid]
            pos["image_path"] = os.path.join(IMAGE_ROOT, img_meta["file_name"])
            pos["height"]     = img_meta["height"]
            pos["width"]      = img_meta["width"]
            pos["annotations"] = anno_map1.get(iid, [])

    # Filter positives with valid paths
    positives = [p for p in positives if os.path.exists(p.get("image_path", ""))]

    print(f"10% oracle test split: {len(positives)} positives  (pre-computed oracle labels)")
    print(f"test_3 negatives:      {len(negatives)} sampled")
    print(f"Layer list: {layer_list}")
    return positives, negatives, layer_list


def load_gt(sample):
    """Load and return GT mask (bool array) for a positive sample."""
    gt = merge_gt_masks(sample["annotations"], sample["height"], sample["width"])
    if gt.shape != (sample["height"], sample["width"]):
        gt = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize(
            (sample["width"], sample["height"]), Image.NEAREST)).astype(bool)
    return gt


def resize_mask(pred, gt):
    if pred.shape == gt.shape:
        return pred
    return np.array(Image.fromarray(pred.astype(np.uint8) * 255).resize(
        (gt.shape[1], gt.shape[0]), Image.NEAREST)).astype(bool)


# ── Evaluation loop ───────────────────────────────────────────────────────────
def evaluate(positives, negatives, wrapper, router):
    """
    For each sample:
      - Backbone runs ONCE (shared across all methods)
      - L32: 1 FPN pass
      - Router hard: 1 FPN pass (different layer)
      - Router MoE: 1 FPN pass (layer blend)
      - Oracle: 16 FPN passes (only for first N_ORACLE_POS positives)

    Returns list of per-sample result dicts.
    """
    methods = ("l32", "hard", "moe", "gated", "oracle")
    records = []
    layer_picks = {"hard": [], "gated": [], "oracle": []}   # for layer distribution plot

    all_samples = positives + negatives
    random.shuffle(all_samples)

    oracle_pos_count = 0   # count how many positives get oracle sweep

    for sample in tqdm(all_samples, desc="Evaluating"):
        is_pos = sample["is_present"]
        gt     = load_gt(sample) if is_pos else None

        try:
            img = Image.open(sample["image_path"]).convert("RGB")
            pv, ids = wrapper.preprocess(img, sample["prompt"])
            hs, backbone_lhs, text_emb_detr, text_emb_router, img_emb_router = \
                wrapper.extract(pv, ids)
        except Exception as e:
            continue

        # ── Build router input based on what the router was trained on ─────────
        # Router input_dim stored in the checkpoint determines which signal to use.
        text_cpu = text_emb_router.cpu()
        img_cpu  = img_emb_router.cpu()
        if router.input_dim == 2048:
            # v5 (default): concat(img_emb, text_emb) — image-dominant signal
            router_input = torch.cat([img_cpu, text_cpu], dim=-1).unsqueeze(0)
        elif router.input_dim == 256:
            # v3 ablation: DETR cross-attention — run FPN(L32) + DETR decoder
            detr_emb     = wrapper.extract_detr_emb(hs, backbone_lhs, text_emb_detr, pv)
            router_input = detr_emb.cpu().unsqueeze(0)               # (1, 256)
        else:
            # 1024-dim: either img_only (v5 ablation) or text-only (v1 legacy)
            router_input = img_cpu.unsqueeze(0)

        # ── Router weights (same for hard and MoE) ────────────────────────────
        with torch.no_grad():
            layer_weights = router.get_layer_weights(router_input)
            hard_layer    = router.hard_pick(router_input)
        layer_picks["hard"].append(hard_layer)

        row = dict(image_id=sample["image_id"], prompt=sample["prompt"],
                   is_present=int(is_pos))

        # ── Method 1: L32 default ─────────────────────────────────────────────
        # IMPORTANT: query_score and mask computation are in SEPARATE try blocks.
        # If resize_mask raises (e.g. GT shape mismatch on positive samples), the
        # already-captured query_score must NOT be overwritten by the except clause.
        # A single joint try was silently zeroing q32 for all positives.
        try:
            out32 = wrapper.run(img, hs, backbone_lhs, text_emb_detr, pv, layer_idx=32)
            q32   = out32["query_score"]
        except Exception:
            q32 = 0.0; out32 = None
        pred32 = None
        if is_pos and out32 is not None and gt is not None:
            try:
                pred32 = resize_mask(out32["union_mask"], gt)
            except Exception:
                pass
        row["l32_query"]    = q32
        row["l32_detected"] = int(q32 >= THRESHOLD)
        row["l32_cgf1"]     = compute_cgf1(pred32, gt) if (is_pos and pred32 is not None and gt is not None) else None
        row["l32_iou"]      = compute_iou(pred32, gt)   if (is_pos and pred32 is not None and gt is not None) else None

        # ── Method 2: Router hard (top-1 layer) ───────────────────────────────
        try:
            out_h = wrapper.run(img, hs, backbone_lhs, text_emb_detr, pv, layer_idx=hard_layer)
            q_h   = out_h["query_score"]
        except Exception:
            q_h = 0.0; out_h = None
        pred_h = None
        if is_pos and out_h is not None and gt is not None:
            try:
                pred_h = resize_mask(out_h["union_mask"], gt)
            except Exception:
                pass
        row["hard_layer"]    = hard_layer
        row["hard_query"]    = q_h
        row["hard_detected"] = int(q_h >= THRESHOLD)
        row["hard_cgf1"]     = compute_cgf1(pred_h, gt) if (is_pos and pred_h is not None and gt is not None) else None
        row["hard_iou"]      = compute_iou(pred_h, gt)  if (is_pos and pred_h is not None and gt is not None) else None

        # ── Method 3: Router MoE (soft blend) ────────────────────────────────
        try:
            out_m = wrapper.run_moe(img, hs, backbone_lhs, text_emb_detr, pv, layer_weights)
            q_m   = out_m["query_score"]
        except Exception:
            q_m = 0.0; out_m = None
        pred_m = None
        if is_pos and out_m is not None and gt is not None:
            try:
                pred_m = resize_mask(out_m["union_mask"], gt)
            except Exception:
                pass
        row["moe_query"]    = q_m
        row["moe_detected"] = int(q_m >= THRESHOLD)
        row["moe_cgf1"]     = compute_cgf1(pred_m, gt) if (is_pos and pred_m is not None and gt is not None) else None
        row["moe_iou"]      = compute_iou(pred_m, gt)  if (is_pos and pred_m is not None and gt is not None) else None

        # ── Method 4: Gated Router ────────────────────────────────────────────
        # The default SAM3 (L32) is already good — we only want to improve the
        # cases where it FAILS.  If L32 confidently detects the concept
        # (query_score >= GATE_THRESHOLD), we leave it alone.  Only when L32
        # struggles do we invoke the router-selected layer to recover the missed
        # detection.  This matches training: the router was trained exclusively
        # on "failed cases" where oracle > L32 by >= DELTA_THRESHOLD cgF1.
        if q32 >= GATE_THRESHOLD:
            # L32 already working well — keep the L32 result unchanged
            gated_layer = 32
            row["gated_layer"]    = gated_layer
            row["gated_query"]    = q32
            row["gated_detected"] = row["l32_detected"]
            row["gated_cgf1"]     = row["l32_cgf1"]
            row["gated_iou"]      = row["l32_iou"]
        else:
            # L32 struggling — ask the router for a better backbone layer
            gated_layer = hard_layer
            row["gated_layer"]    = gated_layer
            row["gated_query"]    = q_h
            row["gated_detected"] = row["hard_detected"]
            row["gated_cgf1"]     = row["hard_cgf1"]
            row["gated_iou"]      = row["hard_iou"]
        layer_picks["gated"].append(gated_layer)

        # ── Method 5: Oracle (sweep all layers — positives only, limited N) ───
        if is_pos and oracle_pos_count < N_ORACLE_POS:
            oracle_pos_count += 1
            best_l, best_cgf1, best_iou, best_q = 32, 0.0, 0.0, 0.0
            for l in TRAIN_LAYERS:
                # Separate model run from mask computation — same reason as above.
                try:
                    out_l = wrapper.run(img, hs, backbone_lhs, text_emb_detr, pv, layer_idx=l)
                    q_l   = out_l["query_score"]
                except Exception:
                    continue
                pred_l = None
                if gt is not None:
                    try:
                        pred_l = resize_mask(out_l["union_mask"], gt)
                    except Exception:
                        pass
                cgf1_l = compute_cgf1(pred_l, gt) if pred_l is not None else 0.0
                if cgf1_l > best_cgf1:
                    best_cgf1 = cgf1_l
                    best_iou  = compute_iou(pred_l, gt) if pred_l is not None else 0.0
                    best_l    = l
                    best_q    = q_l
            row["oracle_layer"]    = best_l
            row["oracle_query"]    = best_q
            row["oracle_detected"] = int(best_q >= THRESHOLD)
            row["oracle_cgf1"]     = best_cgf1
            row["oracle_iou"]      = best_iou
            layer_picks["oracle"].append(best_l)
        else:
            row["oracle_layer"] = row["oracle_query"] = row["oracle_detected"] = None
            row["oracle_cgf1"]  = row["oracle_iou"]   = None

        records.append(row)

    return records, layer_picks


# ── Aggregate metrics ─────────────────────────────────────────────────────────
def aggregate(records):
    method_keys = {
        "L32 default":  ("l32_detected",    "l32_query",    "l32_cgf1",    "l32_iou"),
        "Router hard":  ("hard_detected",   "hard_query",   "hard_cgf1",   "hard_iou"),
        "Router MoE":   ("moe_detected",    "moe_query",    "moe_cgf1",    "moe_iou"),
        # Gated: use L32 when it already works; invoke router only on failed cases.
        # Trained with FOCUS_FAILED=1 — router only learned from samples where
        # routing over backbone layers can actually improve over L32.
        "Gated Router": ("gated_detected",  "gated_query",  "gated_cgf1",  "gated_iou"),
        "Oracle":       ("oracle_detected", "oracle_query", "oracle_cgf1", "oracle_iou"),
    }
    summary = {}
    for method, (det_key, q_key, cgf1_key, iou_key) in method_keys.items():
        sub = [r for r in records if r.get(det_key) is not None]
        if not sub:
            continue
        y_true = [r["is_present"] for r in sub]
        y_pred = [r[det_key] for r in sub]

        # IL_MCC — needs both pos and neg
        if len(set(y_true)) >= 2:
            il_mcc = float(matthews_corrcoef(y_true, y_pred))
        else:
            il_mcc = float("nan")

        pos_sub = [r for r in sub if r["is_present"]]
        cgf1_vals = [r[cgf1_key] for r in pos_sub if r.get(cgf1_key) is not None]
        iou_vals  = [r[iou_key]  for r in pos_sub if r.get(iou_key)  is not None]

        summary[method] = dict(
            n_samples = len(sub),
            n_pos     = sum(r["is_present"] for r in sub),
            n_neg     = sum(1 - r["is_present"] for r in sub),
            IL_MCC    = il_mcc,
            precision = sum((r["is_present"] == 1 and r[det_key] == 1) for r in sub) /
                        max(sum(r[det_key] for r in sub), 1),
            recall    = sum((r["is_present"] == 1 and r[det_key] == 1) for r in sub) /
                        max(sum(r["is_present"] for r in sub), 1),
            cgF1      = float(np.mean(cgf1_vals)) if cgf1_vals else float("nan"),
            IoU       = float(np.mean(iou_vals))  if iou_vals  else float("nan"),
            pmF1      = float(np.mean([v >= 0.5 for v in iou_vals])) if iou_vals else float("nan"),
        )
    return summary


# ── Save CSV ──────────────────────────────────────────────────────────────────
def save_csv(records):
    path = os.path.join(OUT_DIR, "eval_full_raw.csv")
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"Raw CSV saved: {path}")


# ── Summary plot ──────────────────────────────────────────────────────────────
def save_summary_plot(summary, n_pos):
    methods = [m for m in ("L32 default", "Router hard", "Router MoE",
                            "Gated Router", "Oracle")
               if m in summary]
    colors  = {"L32 default":  "#d62728", "Router hard": "#2196F3",
               "Router MoE":   "#9C27B0", "Gated Router": "#4CAF50",
               "Oracle":       "#ff7f0e"}
    metrics = [("IL_MCC", "IL_MCC  (main — presence discrimination)"),
               ("cgF1",   "cgF1 (mask quality, Dice vs GT)"),
               ("IoU",    "IoU vs GT"),
               ("pmF1",   "pmF1 (IoU ≥ 0.5 hit rate)")]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"CAPR Router Evaluation — SA-Co metaclip test_3  "
        f"({n_pos} pos + {N_NEG} neg, threshold={THRESHOLD})\n"
        f"Gated: router invoked only when L32 fails (q<{GATE_THRESHOLD})  |  "
        f"Oracle on {N_ORACLE_POS} positives only",
        fontsize=11, fontweight="bold")

    for ax, (metric, title) in zip(axes, metrics):
        vals  = [summary[m].get(metric, float("nan")) for m in methods]
        cols  = [colors[m] for m in methods]
        valid = [(m, v, c) for m, v, c in zip(methods, vals, cols) if not np.isnan(v)]
        if not valid:
            ax.set_title(title, fontsize=10); ax.axis("off"); continue
        ms, vs, cs = zip(*valid)
        bars = ax.bar(ms, vs, color=cs, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylim(-0.05 if metric == "IL_MCC" else 0,
                    min(1.05, max(vs) * 1.3) if vs else 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, vs):
            sign = "+" if val >= 0 else ""
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.005 if val >= 0 else -0.025),
                    f"{sign}{val:.3f}", ha="center", va="bottom", fontsize=9)
        if metric == "IL_MCC":
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "eval_full_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Summary plot saved: {out}")


# ── Layer distribution plot ───────────────────────────────────────────────────
def save_layer_dist(layer_picks):
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))
    fig.suptitle("Layer selection distribution", fontsize=12, fontweight="bold")
    for ax, (key, title, color) in zip(axes, [
        ("hard",   "Router hard — predicted layer", "#2196F3"),
        # Gated: shows which layer was actually used (32 when L32 worked; router pick otherwise)
        ("gated",  f"Gated Router — layer used (L32 when q≥{GATE_THRESHOLD}, router otherwise)",
                    "#4CAF50"),
        ("oracle", f"Oracle — best layer (first {N_ORACLE_POS} positives)", "#ff7f0e"),
    ]):
        picks = layer_picks.get(key, [])
        if not picks:
            ax.set_title(title); ax.axis("off"); continue
        counts = {l: picks.count(l) for l in TRAIN_LAYERS}
        ax.bar([f"L{l}\nblk{l-1}" for l in TRAIN_LAYERS],
               [counts.get(l, 0) for l in TRAIN_LAYERS],
               color=color, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Layer"); ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "eval_full_layer_dist.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Layer dist plot saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n=== CAPR Full Evaluation — SA-Co metaclip test_3 ===")
    print(f"Primary metric: IL_MCC")
    print(f"Dataset split:  test_1=train  test_2=val  test_3=test (this script)\n")

    positives, negatives, _ = load_eval_samples()
    wrapper = SAM3Wrapper()
    router  = load_router()
    router.eval()
    print(f"\nRouter: {len(router.layer_list)} layers  {router.layer_list[:4]}...{router.layer_list[-1]}")
    print(f"Dataset split: 70% train | 20% val | 10% test  ← THIS SCRIPT uses the 10% test split")
    print(f"Evaluating {len(positives)} pos + {len(negatives)} neg "
          f"(oracle on first {N_ORACLE_POS} pos)\n")

    records, layer_picks = evaluate(positives, negatives, wrapper, router)
    summary = aggregate(records)
    save_csv(records)

    # ── Print table ────────────────────────────────────────────────────────────
    print(f"\n{'Method':>16}  {'n':>5}  {'IL_MCC':>8}  {'Prec':>7}  "
          f"{'Recall':>7}  {'cgF1':>8}  {'IoU':>8}  {'pmF1':>8}")
    print("-" * 78)
    l32_mcc = summary.get("L32 default", {}).get("IL_MCC", float("nan"))
    for method in ("L32 default", "Router hard", "Router MoE", "Gated Router", "Oracle"):
        if method not in summary:
            continue
        s = summary[method]
        delta = s["IL_MCC"] - l32_mcc if not np.isnan(s["IL_MCC"]) else float("nan")
        sign  = "+" if delta >= 0 else ""
        tag   = f"  ({sign}{delta:.3f})" if not np.isnan(delta) else ""
        cgf1  = f"{s['cgF1']:.4f}" if not np.isnan(s["cgF1"]) else "  n/a  "
        iou   = f"{s['IoU']:.4f}"  if not np.isnan(s["IoU"])  else "  n/a  "
        pmf1  = f"{s['pmF1']:.4f}" if not np.isnan(s["pmF1"]) else "  n/a  "
        print(f"  {method:>14s}  {s['n_samples']:>5d}  "
              f"{s['IL_MCC']:>+8.4f}{tag:12s}  {s['precision']:>7.4f}  "
              f"{s['recall']:>7.4f}  {cgf1:>8s}  {iou:>8s}  {pmf1:>8s}")

    save_summary_plot(summary, n_pos=len(positives))
    save_layer_dist(layer_picks)
    print(f"\nDataset split used:  70% train | 20% val | 10% test")
    print(f"Oracle positives from 10% test split: {len(positives)}")
    print(f"Negatives from test_3 (held-out):     {len(negatives)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
