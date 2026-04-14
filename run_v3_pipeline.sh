#!/usr/bin/env bash
# ============================================================
# CAPR v3 Pipeline — DETR cross-attention router
#
# Router input: SAM3's own text-image cross-attention output (256-dim)
#   - Runs FPN(L32) + DETR decoder for each sample
#   - Uses last decoder layer mean over 200 queries as routing signal
#   - This is SAM3's internal text-image fusion, not raw embeddings
#
# Steps:
#   1. extract_detr_embs.py  — FPN(L32)+DETR for 1368 oracle samples,
#                              saves detr_embs.npy (256-dim)  (~15 min)
#   2. train_router.py       — auto-detects detr_embs.npy, trains with
#                              256-dim input, 150 epochs              (~2 min)
#   3. eval_router_full.py   — at inference: two-pass routing
#                              (pass1: FPN(L32)+DETR for routing emb,
#                               pass2: FPN(Lk)+DETR for final output) (~30 min)
#
# Output log: results/pipeline_v3.log
# ============================================================
set -euo pipefail

CAPR_DIR="/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/capr_clean"
LOG="$CAPR_DIR/results/pipeline_v3.log"
mkdir -p "$CAPR_DIR/results"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*" | tee -a "$LOG"; }

cd "$CAPR_DIR"

log "=============================================="
log "CAPR v3 Pipeline — DETR cross-attention router"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
log "Router input: SAM3 DETR decoder last layer (256-dim text-image fusion)"
log "=============================================="

log "Step 1/3 — Extracting DETR cross-attention embeddings..."
log "  Expected runtime: ~15 min"
python experiments/extract_detr_embs.py 2>&1 | tee -a "$LOG"
log "Step 1 DONE"

log ""
log "Step 2/3 — Training CAPR router v3 (DETR 256-dim, 150 epochs)..."
log "  Expected runtime: ~2 min"
python experiments/train_router.py 2>&1 | tee -a "$LOG"
log "Step 2 DONE"

log ""
log "Step 3/3 — Full evaluation (two-pass routing, IL_MCC primary metric)..."
log "  Expected runtime: ~30 min"
python experiments/eval_router_full.py 2>&1 | tee -a "$LOG"
log "Step 3 DONE"

log ""
log "=============================================="
log "ALL STEPS COMPLETED"
log "Results in: $CAPR_DIR/results/"
log "  detr_embs.npy             — DETR cross-attn embeddings (256-dim)"
log "  capr_router_weights.pt    — trained router checkpoint (v3)"
log "  router_training_curve.png — training curve"
log "  eval_full_raw.csv         — per-sample results"
log "  eval_full_summary.png     — IL_MCC / cgF1 / IoU / pmF1 bar chart"
log "  eval_full_layer_dist.png  — router layer selection distribution"
log "=============================================="
