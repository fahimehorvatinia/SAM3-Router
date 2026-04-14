#!/usr/bin/env bash
# ============================================================
# CAPR Cross-Dataset Evaluation
#
# Evaluates the trained CAPR router (from results/capr_router_weights.pt)
# on all available SA-Co datasets:
#   - metaclip         (original training domain — baseline reference)
#   - attributes       (concept grounding with attribute descriptions)
#   - crowded          (crowded scenes — harder for presence detection)
#   - wiki-food&drink  (food and drink concepts from Wikipedia)
#   - wiki-sports_equipment  (sports equipment concepts from Wikipedia)
#
# Tests zero-shot generalization: the router was trained on metaclip,
# and we evaluate without retraining on the new domains.
#
# Usage:
#   cd capr_clean
#   bash run_crossdataset.sh
# ============================================================
set -euo pipefail

LOG="results/crossdataset/run.log"
mkdir -p results/crossdataset
echo "[$(date)] CAPR cross-dataset evaluation" | tee "$LOG"
echo "Router: results/capr_router_weights.pt" | tee -a "$LOG"
echo "GATE_THRESHOLD=0.5   N_POS=200   N_NEG=500" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Datasets to evaluate (sa-1b excluded — images not locally available)
DATASETS=(
    "metaclip"
    "attributes"
    "crowded"
    "wiki-food&drink"
    "wiki-sports_equipment"
)

for DS in "${DATASETS[@]}"; do
    echo "────────────────────────────────────────" | tee -a "$LOG"
    echo "[$(date)] Evaluating: $DS" | tee -a "$LOG"
    DATASET="$DS" GATE_THRESHOLD=0.5 N_POS=200 N_NEG=500 \
        python experiments/eval_crossdataset.py 2>&1 | tee -a "$LOG"
    echo "[$(date)] Done: $DS" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

echo "════════════════════════════════════════" | tee -a "$LOG"
echo "[$(date)] All datasets done." | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── Print combined summary table ─────────────────────────────────────────────
echo "COMBINED RESULTS SUMMARY" | tee -a "$LOG"
echo "========================" | tee -a "$LOG"
printf "%-22s  %-14s  %-8s  %-6s  %-6s  %-6s\n" \
    "Dataset" "Method" "IL_MCC" "Recall" "cgF1" "IoU" | tee -a "$LOG"
echo "─────────────────────────────────────────────────────────────────────" | tee -a "$LOG"
for DS in "${DATASETS[@]}"; do
    SUMMARY="results/crossdataset/${DS}_summary.txt"
    if [ -f "$SUMMARY" ]; then
        while IFS= read -r line; do
            if echo "$line" | grep -qE "^.*(L32|Router|Gated)"; then
                method=$(echo "$line" | awk '{print $1, $2}')
                printf "%-22s  %-14s  %s\n" "$DS" "$method" "$line" | tee -a "$LOG"
            fi
        done < "$SUMMARY"
    fi
done
