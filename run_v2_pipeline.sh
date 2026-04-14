#!/usr/bin/env bash
# ============================================================
# CAPR v2 Pipeline — text+image router
#
# Steps:
#   1. extract_img_embs.py   — backbone-only pass for 1368 existing
#                              oracle samples, saves img_embs.npy (~10 min)
#   2. train_router.py       — train router on concat(text, img) input
#                              (2048-dim), 150 epochs (~2 min)
#   3. eval_router_full.py   — evaluate on 10% test split + test_3 negatives
#                              primary metric: IL_MCC (~30 min)
#
# Output log: results/pipeline_v2.log
# ============================================================
set -euo pipefail

CAPR_DIR="/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/capr_clean"
LOG="$CAPR_DIR/results/pipeline_v2.log"
mkdir -p "$CAPR_DIR/results"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*" | tee -a "$LOG"; }

cd "$CAPR_DIR"

log "=============================================="
log "CAPR v2 Pipeline — text+image router"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
log "=============================================="

log "Step 1/3 — Extracting image embeddings for existing oracle samples..."
log "  Expected runtime: ~10 min"
python experiments/extract_img_embs.py 2>&1 | tee -a "$LOG"
log "Step 1 DONE"

log ""
log "Step 2/3 — Training CAPR router v2 (text+image, 150 epochs)..."
log "  Expected runtime: ~2 min"
python experiments/train_router.py 2>&1 | tee -a "$LOG"
log "Step 2 DONE"

log ""
log "Step 3/3 — Full evaluation on test_3 (IL_MCC primary metric)..."
log "  Expected runtime: ~30 min"
python experiments/eval_router_full.py 2>&1 | tee -a "$LOG"
log "Step 3 DONE"

log ""
log "=============================================="
log "ALL STEPS COMPLETED"
log "Results in: $CAPR_DIR/results/"
log "  img_embs.npy              — image embeddings for training"
log "  capr_router_weights.pt    — trained router checkpoint (v2)"
log "  router_training_curve.png — training curve"
log "  eval_full_raw.csv         — per-sample results"
log "  eval_full_summary.png     — IL_MCC / cgF1 / IoU / pmF1 bar chart"
log "  eval_full_layer_dist.png  — router layer selection distribution"
log "=============================================="
