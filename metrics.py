"""
SAM3 paper metrics: IL_MCC, cgF1, pmF1.

IL_MCC  — Instance-Level Matthews Correlation Coefficient (concept recognition).
           Requires both positive and negative pairs.
           Threshold presence_score at 0.5 → binary prediction.

cgF1    — Concept Grounding F1.
           Dice/F1 between predicted mask and GT mask, averaged over positive samples.
           Measures segmentation quality given the concept is present.

pmF1    — Positive Micro F1 (called positive_micro_F1 in SAM3 code).
           Among positive samples: fraction where IoU(pred, GT) >= 0.5.
           Measures how often the model finds a good mask when the concept is present.
"""
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix


PRESENCE_THRESHOLD = 0.5
IOU_THRESHOLD      = 0.5   # for pmF1


def compute_il_mcc(y_true: list, presence_scores: list) -> dict:
    """
    IL_MCC: Matthews Correlation Coefficient on binary presence prediction.

    Args:
        y_true          : list of 1 (positive) or 0 (negative)
        presence_scores : list of floats in [0, 1]

    Returns dict with il_mcc, precision, recall, f1, tp, tn, fp, fn
    """
    y_pred = [1 if s > PRESENCE_THRESHOLD else 0 for s in presence_scores]
    mcc    = float(matthews_corrcoef(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "il_mcc":    round(mcc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
    }


def compute_cgf1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    cgF1: Dice/F1 between predicted and GT binary masks (single sample).
    Average over positive samples to get the dataset-level cgF1.
    """
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)
    intersection = (pred & gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Intersection-over-Union between two binary masks."""
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)
    intersection = (pred & gt).sum()
    union        = (pred | gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_pmf1(ious: list) -> float:
    """
    pmF1: fraction of positive samples where IoU(pred, GT) >= IOU_THRESHOLD.
    (positive_micro_F1 in the SAM3 codebase)
    """
    if not ious:
        return 0.0
    return float(np.mean([iou >= IOU_THRESHOLD for iou in ious]))


def decode_rle(seg: dict, h: int, w: int) -> np.ndarray:
    """Decode a pycocotools RLE segmentation dict to a binary mask."""
    import pycocotools.mask as maskUtils
    if isinstance(seg.get("counts"), list):
        rle = maskUtils.frPyObjects(seg, h, w)
    else:
        rle = seg
    return maskUtils.decode(rle).astype(bool)


def merge_gt_masks(annotations: list, h: int, w: int) -> np.ndarray:
    """Merge all GT annotation masks for one image into a single binary mask."""
    merged = np.zeros((h, w), dtype=bool)
    for ann in annotations:
        seg = ann.get("segmentation", {})
        if isinstance(seg, dict) and "counts" in seg:
            merged |= decode_rle(seg, h, w)
    return merged
