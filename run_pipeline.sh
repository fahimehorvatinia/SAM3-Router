#!/usr/bin/env bash
# ============================================================
# CAPR Full Pipeline — runs unattended
#
# Dataset split (70 / 20 / 10):
#   70% → router training data  (oracle labels collected, used in train_router.py)
#   20% → router validation     (used for early-stopping / val loss in train_router.py)
#   10% → held-out test         (eval_router_full.py, IL_MCC primary metric)
#
# Steps:
#   1. collect_oracle_layers.py  — sweep 16 layers on 1500 SA-Co test_1
#                                  positives, save text_embs + cgF1_matrix
#                                  + writes 70/20/10 split indices to meta.json
#   2. train_router.py           — train CAPRRouter on 70% split
#                                  (KL + CE loss, 150 epochs, val on 20% split)
#   3. eval_router_full.py       — evaluate on 10% test split + test_3 negatives
#                                  primary metric: IL_MCC
#
# Output log: results/pipeline_run.log
# ============================================================

set -euo pipefail

CAPR_DIR="/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/capr_clean"
LOG="$CAPR_DIR/results/pipeline_run.log"
mkdir -p "$CAPR_DIR/results"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*" | tee -a "$LOG"
}

cd "$CAPR_DIR"

log "=============================================="
log "CAPR Pipeline starting"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
log "=============================================="

# ── Step 1: Collect oracle layer data ──────────────────────────────────────────
log "Step 1/3 — Collecting oracle layers (1500 positives from test_1)..."
log "  Expected runtime: ~30 min"
python experiments/collect_oracle_layers.py 2>&1 | tee -a "$LOG"
log "Step 1 DONE"

# ── Step 2: Train the router ────────────────────────────────────────────────────
log ""
log "Step 2/3 — Training CAPR router (150 epochs)..."
log "  Expected runtime: ~2 min"
python experiments/train_router.py 2>&1 | tee -a "$LOG"
log "Step 2 DONE"

# ── Step 3: Full evaluation on test_3 ──────────────────────────────────────────
log ""
log "Step 3/3 — Full evaluation on test_3 (500 pos + 500 neg, IL_MCC primary)..."
log "  Expected runtime: ~30 min"
python experiments/eval_router_full.py 2>&1 | tee -a "$LOG"
log "Step 3 DONE"

log ""
log "=============================================="
log "ALL STEPS COMPLETED"
log "Results in: $CAPR_DIR/results/"
log "  router_training_data/   — collected oracle labels"
log "  capr_router_weights.pt  — trained router checkpoint"
log "  router_training_curve.png"
log "  eval_full_raw.csv       — per-sample results"
log "  eval_full_summary.png   — IL_MCC / cgF1 / IoU / pmF1 bar chart"
log "  eval_full_layer_dist.png — router layer selection distribution"
log "=============================================="
