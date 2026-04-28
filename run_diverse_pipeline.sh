#!/usr/bin/env bash
# =============================================================================
# Diverse Training Pipeline — Reviewer Experiment 3
#
# Runs three steps sequentially:
#   Step 1 — collect oracle layer data from 6 SA-Co domains (~5-8 h on GPU)
#   Step 2 — train MLP router on combined diverse data        (~2 min)
#   Step 3 — evaluate diverse-trained router on MetaCLIP test (~7 min)
#
# All output goes to:   results/diverse_pipeline.log
# Checkpoint saved to:  results/capr_router_weights_diverse.pt
# Eval CSV saved to:    results/eval_full_raw.csv  (overwritten by each eval)
#   -> to preserve, we also copy it to results/eval_diverse_raw.csv
#
# Submit with:
#   nohup bash run_diverse_pipeline.sh > results/diverse_pipeline.log 2>&1 &
#   echo "PID: $!"
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

LOG=results/diverse_pipeline.log
DIVDIR=results/router_training_data_diverse

echo "================================================================"
echo " Diverse Training Pipeline — started at $(date)"
echo " Working dir: $(pwd)"
echo " GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "================================================================"

# ─── Step 1: Collect oracle labels from 6 domains ─────────────────────────
echo ""
echo "────────────────────────────────────────────────────────────────"
echo " STEP 1/3 — Oracle collection (6 domains × 1500 samples)"
echo " Domains: metaclip, attributes, crowded, wiki-food, wiki-sports, wiki-common1k"
echo " Estimated time: 5–8 hours"
echo " Started: $(date)"
echo "────────────────────────────────────────────────────────────────"

python experiments/collect_oracle_diverse.py

echo ""
echo " Step 1 complete at $(date)"
echo " Files in $DIVDIR:"
ls -lh "$DIVDIR/"

# ─── Step 2: Train diverse router (concat mode on img+text embeddings) ────
echo ""
echo "────────────────────────────────────────────────────────────────"
echo " STEP 2/3 — Train router on diverse data"
echo " EMB_MODE=concat (2048-dim img+text), DATA_DIR=$DIVDIR"
echo " Expected time: ~5 min"
echo " Started: $(date)"
echo "────────────────────────────────────────────────────────────────"

DATA_DIR="$(pwd)/$DIVDIR" EMB_MODE=concat python experiments/train_router.py

echo ""
echo " Step 2 complete at $(date)"
echo " Checkpoint: results/capr_router_weights_diverse.pt"

# ─── Also train attention router on diverse data ──────────────────────────
# (Optional: comment out if you only want MLP results)
# First need to extract full DETR sequences for diverse samples.
# This is skipped here to keep runtime manageable.
# To enable: uncomment the block below.
#
# echo ""
# echo " STEP 2b — Extract full DETR embeddings for diverse data"
# DATA_DIR="$(pwd)/$DIVDIR" python experiments/extract_detr_embs_full.py
# DATA_DIR="$(pwd)/$DIVDIR" EMB_MODE=detr_full python experiments/train_router.py
# ROUTER_WEIGHTS=results/capr_router_weights_attn_diverse.pt \
#   python experiments/eval_router_full.py 2>&1 | tee -a results/eval_attn_diverse.log

# ─── Step 3: Evaluate diverse-trained router ──────────────────────────────
echo ""
echo "────────────────────────────────────────────────────────────────"
echo " STEP 3/3 — Evaluate diverse router on MetaCLIP test_3"
echo " ROUTER_WEIGHTS=results/capr_router_weights_diverse.pt"
echo " Expected time: ~7 min"
echo " Started: $(date)"
echo "────────────────────────────────────────────────────────────────"

ROUTER_WEIGHTS=results/capr_router_weights_diverse.pt \
  python experiments/eval_router_full.py 2>&1 | tee results/eval_diverse_raw_summary.txt

# Save a copy of the raw CSV under a distinct name
cp results/eval_full_raw.csv results/eval_diverse_raw.csv 2>/dev/null || true

# ─── Cross-domain eval with diverse router ───────────────────────────────
echo ""
echo "────────────────────────────────────────────────────────────────"
echo " BONUS — Cross-domain eval with diverse router"
echo " Started: $(date)"
echo "────────────────────────────────────────────────────────────────"

for DOMAIN in attributes crowded "wiki-food&drink" wiki-sports_equipment; do
    echo "  → $DOMAIN"
    DATASET=$DOMAIN \
    ROUTER_WEIGHTS=results/capr_router_weights_diverse.pt \
    GATE_THRESHOLD=0.5 \
      python experiments/eval_crossdataset.py 2>&1 | tail -15
done

# ─── Final summary ─────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " ALL STEPS COMPLETE at $(date)"
echo ""
echo " Results to review:"
echo "   results/capr_router_weights_diverse.pt   — diverse-trained MLP router"
echo "   results/eval_diverse_raw.csv             — per-sample eval (MetaCLIP test)"
echo "   results/eval_diverse_raw_summary.txt     — summary table"
echo "   results/crossdataset/*_summary.txt       — cross-domain results"
echo ""
echo " Compare in paper:"
echo "   Gated MLP (MetaCLIP-only) vs Gated MLP (diverse)"
echo "   Look for improvement in cgF1, recall, and especially IL_MCC"
echo "================================================================"
