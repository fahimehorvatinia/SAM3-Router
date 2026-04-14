#!/usr/bin/env bash
# ============================================================
# CAPR v5 Pipeline — Improved router: image+text signal, balanced training
#
# What changed vs v4:
#   1. Router input: concat(img_emb, text_emb) = 2048-dim (stronger image signal)
#      The optimal backbone layer depends more on image content than concept text.
#      Using both provides a richer routing signal than DETR (256-dim) alone.
#   2. DELTA_THRESHOLD lowered to 0.02: 662 failed-case training samples vs 467.
#      More data + better coverage of all 16 layers.
#   3. Class-balanced CE loss: prevents router collapsing to predicting L31/L32
#      by upweighting under-represented early layers (L17-L28).
#   4. 300 epochs + AdamW + warmup-cosine schedule: better convergence.
#   5. GATE_THRESHOLD lowered to 0.4: the more accurate router can be invoked
#      more aggressively without hurting clean cases.
#
# Steps:
#   Step 1: Retrain router    (oracle data already collected, skip Step 0)
#   Step 2: Run full eval
# ============================================================
set -euo pipefail

LOG="results/pipeline_v5.log"
mkdir -p results

echo "[$(date)] CAPR v5 pipeline — improved router" | tee "$LOG"
echo "  Embedding: concat(img_emb, text_emb) = 2048-dim" | tee -a "$LOG"
echo "  DELTA_THRESHOLD=0.02  (662 failed cases)" | tee -a "$LOG"
echo "  Class-weighted CE, 300 epochs, AdamW + warmup-cosine LR" | tee -a "$LOG"
echo "  GATE_THRESHOLD=0.4 for gated-router inference" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── Step 1: Train router with improved signal ────────────────────────────────
echo "[$(date)] Step 1: Training router (concat img+text, 300 epochs)" | tee -a "$LOG"
EMB_MODE=concat \
DELTA_THRESHOLD=0.02 \
USE_CLASS_WEIGHTS=1 \
  python experiments/train_router.py 2>&1 | tee -a "$LOG"
echo "[$(date)] Step 1 done." | tee -a "$LOG"

# ── Step 2: Evaluate with gated router ──────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[$(date)] Step 2: Evaluating (GATE_THRESHOLD=0.4)" | tee -a "$LOG"
GATE_THRESHOLD=0.4 \
  python experiments/eval_router_full.py 2>&1 | tee -a "$LOG"
echo "[$(date)] Step 2 done." | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[$(date)] CAPR v5 pipeline complete." | tee -a "$LOG"
