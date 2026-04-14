#!/usr/bin/env bash
# ============================================================
# CAPR v4 Pipeline — Failed-cases-focused router + Gated inference
#
# Key design principle:
#   The default SAM3 (L32) is already good on easy samples.
#   We only care about improving cases where L32 FAILS.
#
# Changes from v3:
#   TRAINING (train_router.py):
#     FOCUS_FAILED=1  — filter oracle data to samples where
#                       max_cgF1 - cgF1_at_L32 >= DELTA_THRESHOLD (0.05).
#                       The router only sees hard cases where routing helps.
#                       This eliminates the L31/L32 mode collapse that plagued v1-v3.
#
#   INFERENCE (eval_router_full.py):
#     Gated Router  — if L32 already detects (query_score >= GATE_THRESHOLD=0.5),
#                     keep L32.  Only invoke the router when L32 is struggling.
#                     This matches the training objective at test time.
#
# Also fixed in this version:
#   - sam3_wrapper.py extract_detr_emb now passes hidden_states=None to avoid
#     in-place tensor corruption that caused recall=0.0 in v3.
#   - save_summary_plot N_POS NameError fixed.
#
# Reuses:  detr_embs.npy from v3 (no need to re-extract)
# Re-runs: train_router.py  (~2 min with focused dataset)
#           eval_router_full.py (~30 min)
#
# Output log: results/pipeline_v4.log
# ============================================================
set -euo pipefail

CAPR_DIR="/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/capr_clean"
LOG="$CAPR_DIR/results/pipeline_v4.log"
mkdir -p "$CAPR_DIR/results"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*" | tee -a "$LOG"; }

cd "$CAPR_DIR"

log "=============================================="
log "CAPR v4 Pipeline — Failed-cases-focused router"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
log "Router input: DETR cross-attention (256-dim, reused from v3)"
log "Training filter: FOCUS_FAILED=1, DELTA_THRESHOLD=0.05"
log "Inference gate:  GATE_THRESHOLD=0.5 (router only on L32-failed samples)"
log "=============================================="

log ""
log "Step 1/2 — Training CAPR router v4 (failed-cases only, 150 epochs)..."
log "  Reusing detr_embs.npy from v3 — no re-extraction needed."
log "  Expected runtime: ~2 min"
FOCUS_FAILED=1 DELTA_THRESHOLD=0.05 \
    python experiments/train_router.py 2>&1 | tee -a "$LOG"
log "Step 1 DONE"

log ""
log "Step 2/2 — Full evaluation (gated routing, IL_MCC primary metric)..."
log "  Gated: use router only when L32 query_score < GATE_THRESHOLD (0.5)"
log "  Expected runtime: ~30 min"
GATE_THRESHOLD=0.5 \
    python experiments/eval_router_full.py 2>&1 | tee -a "$LOG"
log "Step 2 DONE"

log ""
log "=============================================="
log "ALL STEPS COMPLETED"
log "Results in: $CAPR_DIR/results/"
log "  capr_router_weights.pt    — trained router (v4, failed-cases only)"
log "  router_training_curve.png — training curve"
log "  eval_full_raw.csv         — per-sample results (adds gated_* columns)"
log "  eval_full_summary.png     — IL_MCC/cgF1/IoU/pmF1: L32 | Hard | MoE | Gated | Oracle"
log "  eval_full_layer_dist.png  — layer selection: hard | gated | oracle"
log "=============================================="
