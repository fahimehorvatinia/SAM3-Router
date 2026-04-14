"""
Cross-dataset evaluation of CAPR router on any SA-Co split.

Evaluates the existing trained router (from results/capr_router_weights.pt)
on a new dataset without requiring oracle labels.  Tests zero-shot
generalization of the router to new domains (attributes, crowded, wiki, etc.)

Usage:
    cd capr_clean
    DATASET=attributes python experiments/eval_crossdataset.py
    DATASET=crowded    python experiments/eval_crossdataset.py
    DATASET=wiki-food&drink   python experiments/eval_crossdataset.py
    DATASET=wiki-sports_equipment python experiments/eval_crossdataset.py

Output:
    results/crossdataset/{DATASET}_raw.csv     — per-sample results
    results/crossdataset/{DATASET}_summary.txt — IL_MCC, cgF1, IoU table
"""
import os, sys, json, random
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam3_wrapper import SAM3Wrapper
from capr_router import load_router
from metrics import compute_cgf1, compute_iou, merge_gt_masks

# ── Config ────────────────────────────────────────────────────────────────────
SACO_ROOT  = "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/saco_gold_data"
IMAGE_ROOT = "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/metaclip-images"
OUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "crossdataset")
os.makedirs(OUT_DIR, exist_ok=True)

DATASET        = os.environ.get("DATASET", "attributes")
N_POS          = int(os.environ.get("N_POS",  "200"))   # positives to evaluate
N_NEG          = int(os.environ.get("N_NEG",  "500"))   # negatives to evaluate
THRESHOLD      = float(os.environ.get("THRESHOLD",  "0.5"))
GATE_THRESHOLD = float(os.environ.get("GATE_THRESHOLD", "0.5"))
SEED           = 77

TRAIN_LAYERS = list(range(17, 33))

# ── Dataset loading ───────────────────────────────────────────────────────────
def load_samples(dataset: str):
    """Load positives and negatives from test_3 of the given dataset."""
    # SA-Co datasets use test_3 for held-out evaluation
    json_path = os.path.join(SACO_ROOT, dataset,
                             f"saco_gold_{dataset}_test_3.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")

    with open(json_path) as f:
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
        entry = dict(
            image_id   = img["id"],
            image_path = path,
            prompt     = img["text_input"],
            height     = img["height"],
            width      = img["width"],
            annotations= annos,
        )
        if annos:
            positives.append(entry)
        else:
            negatives.append(entry)

    print(f"Dataset '{dataset}' (test_3): "
          f"{len(positives)} positives  {len(negatives)} negatives available")

    rng = random.Random(SEED)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    return positives[:N_POS], negatives[:N_NEG]


# ── Utilities ─────────────────────────────────────────────────────────────────
def load_gt(sample):
    try:
        gt = merge_gt_masks(sample["annotations"], sample["height"], sample["width"])
        if gt.shape != (sample["height"], sample["width"]):
            gt = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize(
                (sample["width"], sample["height"]), Image.NEAREST)).astype(bool)
        return gt
    except Exception:
        return None


def resize_mask(pred, gt):
    if pred.shape == gt.shape:
        return pred
    return np.array(Image.fromarray(pred.astype(np.uint8) * 255).resize(
        (gt.shape[1], gt.shape[0]), Image.NEAREST)).astype(bool)


def compute_il_mcc(rows, method_detected_key):
    tp = sum(1 for r in rows if r["is_pos"] and r[method_detected_key])
    tn = sum(1 for r in rows if not r["is_pos"] and not r[method_detected_key])
    fp = sum(1 for r in rows if not r["is_pos"] and r[method_detected_key])
    fn = sum(1 for r in rows if r["is_pos"] and not r[method_detected_key])
    denom = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    return (tp*tn - fp*fn) / denom if denom > 0 else 0.0


# ── Main evaluation ───────────────────────────────────────────────────────────
def evaluate(dataset: str):
    print(f"\n{'='*60}")
    print(f"CAPR Cross-Dataset Evaluation — {dataset}")
    print(f"GATE_THRESHOLD={GATE_THRESHOLD}  N_POS={N_POS}  N_NEG={N_NEG}")
    print(f"{'='*60}\n")

    positives, negatives = load_samples(dataset)

    # ── Load models ──────────────────────────────────────────────────────────
    wrapper = SAM3Wrapper()
    router  = load_router()
    router.eval()

    all_samples = [(s, True) for s in positives] + [(s, False) for s in negatives]
    rows = []

    for sample, is_pos in tqdm(all_samples, desc=f"Evaluating {dataset}"):
        row = {"image_id": sample["image_id"], "prompt": sample["prompt"],
               "is_pos": is_pos}

        gt = load_gt(sample) if is_pos else None

        try:
            img = Image.open(sample["image_path"]).convert("RGB")
            pv, ids = wrapper.preprocess(img, sample["prompt"])
            hs, backbone_lhs, text_emb_detr, text_emb_router, img_emb_router = \
                wrapper.extract(pv, ids)
        except Exception:
            continue

        # Build router input
        text_cpu = text_emb_router.cpu()
        img_cpu  = img_emb_router.cpu()
        if router.input_dim == 2048:
            router_input = torch.cat([img_cpu, text_cpu], dim=-1).unsqueeze(0)
        elif router.input_dim == 256:
            detr_emb     = wrapper.extract_detr_emb(hs, backbone_lhs, text_emb_detr, pv)
            router_input = detr_emb.cpu().unsqueeze(0)
        else:
            router_input = img_cpu.unsqueeze(0)

        hard_layer = router.hard_pick(router_input)
        layer_weights = router.get_layer_weights(router_input)

        # ── L32 default ───────────────────────────────────────────────────────
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
        row["l32_cgf1"]     = compute_cgf1(pred32, gt) if pred32 is not None else None
        row["l32_iou"]      = compute_iou(pred32, gt)  if pred32 is not None else None

        # ── Router hard ────────────────────────────────────────────────────────
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
        row["hard_cgf1"]     = compute_cgf1(pred_h, gt) if pred_h is not None else None
        row["hard_iou"]      = compute_iou(pred_h, gt)  if pred_h is not None else None

        # ── Router MoE ─────────────────────────────────────────────────────────
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
        row["moe_cgf1"]     = compute_cgf1(pred_m, gt) if pred_m is not None else None
        row["moe_iou"]      = compute_iou(pred_m, gt)  if pred_m is not None else None

        # ── Gated Router ──────────────────────────────────────────────────────
        if q32 >= GATE_THRESHOLD:
            row["gated_layer"]    = 32
            row["gated_query"]    = q32
            row["gated_detected"] = row["l32_detected"]
            row["gated_cgf1"]     = row["l32_cgf1"]
            row["gated_iou"]      = row["l32_iou"]
        else:
            row["gated_layer"]    = hard_layer
            row["gated_query"]    = q_h
            row["gated_detected"] = row["hard_detected"]
            row["gated_cgf1"]     = row["hard_cgf1"]
            row["gated_iou"]      = row["hard_iou"]

        rows.append(row)

    # ── Aggregate results ────────────────────────────────────────────────────
    pos_rows = [r for r in rows if r["is_pos"]]
    neg_rows = [r for r in rows if not r["is_pos"]]
    n_pos = len(pos_rows)
    n_neg = len(neg_rows)
    n_total = len(rows)

    def agg(method_detected, method_cgf1, method_iou):
        tp = sum(r[method_detected] for r in pos_rows)
        fp = sum(r[method_detected] for r in neg_rows)
        fn = n_pos - tp
        tn = n_neg - fp
        recall    = tp / n_pos if n_pos else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        denom     = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
        mcc       = (tp*tn - fp*fn) / denom if denom > 0 else 0.0
        vals_cgf1 = [r[method_cgf1] for r in pos_rows if r.get(method_cgf1) is not None]
        vals_iou  = [r[method_iou]  for r in pos_rows if r.get(method_iou)  is not None]
        mean_cgf1 = float(np.mean(vals_cgf1)) if vals_cgf1 else float("nan")
        mean_iou  = float(np.mean(vals_iou))  if vals_iou  else float("nan")
        return mcc, precision, recall, mean_cgf1, mean_iou

    methods = [
        ("L32 default",  "l32_detected",   "l32_cgf1",   "l32_iou"),
        ("Router hard",  "hard_detected",  "hard_cgf1",  "hard_iou"),
        ("Router MoE",   "moe_detected",   "moe_cgf1",   "moe_iou"),
        ("Gated Router", "gated_detected", "gated_cgf1", "gated_iou"),
    ]

    print(f"\nDataset: {dataset}  ({n_pos} pos + {n_neg} neg = {n_total} total)\n")
    header = f"{'Method':>14}  {'IL_MCC':>8}  {'Prec':>6}  {'Recall':>6}  {'cgF1':>6}  {'IoU':>6}"
    print(header)
    print("-" * len(header))
    results = {}
    for label, det_key, cgf1_key, iou_key in methods:
        mcc, prec, rec, cgf1, iou = agg(det_key, cgf1_key, iou_key)
        results[label] = dict(IL_MCC=mcc, Prec=prec, Recall=rec, cgF1=cgf1, IoU=iou)
        print(f"{label:>14}  {mcc:>+8.4f}  {prec:>6.4f}  {rec:>6.4f}  "
              f"{cgf1:>6.4f}  {iou:>6.4f}")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    import csv
    csv_path = os.path.join(OUT_DIR, f"{dataset}_raw.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nRaw CSV: {csv_path}")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_path = os.path.join(OUT_DIR, f"{dataset}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"N_POS={n_pos}  N_NEG={n_neg}  GATE_THRESHOLD={GATE_THRESHOLD}\n\n")
        f.write(f"{'Method':>14}  {'IL_MCC':>8}  {'Prec':>6}  {'Recall':>6}  "
                f"{'cgF1':>6}  {'IoU':>6}\n")
        f.write("-" * 60 + "\n")
        for label, _ in [(m[0], None) for m in methods]:
            r = results[label]
            f.write(f"{label:>14}  {r['IL_MCC']:>+8.4f}  {r['Prec']:>6.4f}  "
                    f"{r['Recall']:>6.4f}  {r['cgF1']:>6.4f}  {r['IoU']:>6.4f}\n")
    print(f"Summary:  {summary_path}")
    return results


if __name__ == "__main__":
    evaluate(DATASET)
