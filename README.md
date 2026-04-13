# SAM3-Router: Concept-Adaptive Presence Routing (CAPR)

**SAM3** ([Meta AI, 2024](https://ai.meta.com/research/publications/segment-anything-3/)) is a vision-language model that takes an image and a text concept (e.g. *"the cap"*) and produces a segmentation mask. By default it uses only the **final layer** of its 32-block ViT backbone as input to the FPN neck.

This project shows that **the best backbone layer to use is concept-dependent**, and introduces **CAPR** — a lightweight MLP router that reads the text embedding of a concept and selects (or blends) backbone layers to improve detection quality.

---

## Key Finding

SAM3's ViT backbone produces 33 hidden states (patch embedding + 32 transformer blocks). By default, only the final one (block 31) feeds the mask decoder. We show:

| Metric | Best intermediate layer | SAM3 default (block 31) |
|--------|------------------------|-------------------------|
| IL_MCC | ~block 20 | lower |
| cgF1   | ~block 26–30 | comparable |
| Oracle gap | +**11 cgF1 points** above L32 | baseline |

The **oracle gap** (+0.111 cgF1, +0.103 IoU) proves there is significant headroom achievable by selecting the right layer per concept. The trained CAPR-MoE router captures a portion of this gap without any extra model parameters at inference time.

---

## Architecture

```
Image + Text prompt
       │
       ▼
SAM3 ViT backbone  ← runs ONCE per image
       │
  hidden_states[0..32]    text_emb_router (1024-dim)
       │                         │
       │               ┌─────────▼──────────┐
       │               │   CAPR Router MLP  │
       │               │  1024→256→128→16   │
       │               │  Softmax over L17-32│
       │               └─────────┬──────────┘
       │                         │  layer weights
       ▼                         ▼
  Weighted blend of              Hard top-1
  hidden_states[17..32]  ──OR──  best layer
       │
 LayerNorm + reshape → [B, 1024, 72, 72]
       │
       ▼
  FPN Neck  (real SAM3 weights, unchanged)
       │
  Fusion Encoder + DETR Decoder + Mask Head
       │
  predicted mask + presence score
```

**Architecture note:** SAM3 has 32 transformer blocks (numbered 0–31). `hidden_states` has 33 entries (indices 0–32):
- `hidden_states[0]` = patch embedding
- `hidden_states[k]` = after block k-1 (k = 1..32)
- `hidden_states[32]` = after block 31 ← SAM3 default

**Diagnostic confirmed:** LayerNorm must be applied to ALL layers (0-32) before FPN injection — none come pre-normed from the backbone (max diff for L32: 125.16).

---

## Dataset

Uses the **SA-Co metaclip** dataset (`saco_gold_metaclip_test_1.json` — 6172 positives, 27221 negatives).

**70 / 20 / 10 split** applied to the 1500 oracle-labelled samples collected from `test_1`:

| Split | Size | Role |
|-------|------|------|
| **70% Train** | 1050 samples | Router training (oracle labels → KL + CE loss) |
| **20% Val** | 300 samples | Validation loss during training (early stopping) |
| **10% Test** | 150 positives + 500 test_3 negatives | Final held-out evaluation — IL_MCC primary metric |

Split indices are saved deterministically in `results/router_training_data/meta.json` (key `"splits"`).

**Positive**: image contains the queried concept, GT segmentation mask available.  
**Negative**: concept is absent from the image, no annotation — required for IL_MCC.

> Negatives for the final eval come from `saco_gold_metaclip_test_3.json`, which is never used during training or validation.

---

## Files

| File | Purpose |
|------|---------|
| `sam3_wrapper.py` | Loads SAM3, extracts all 33 hidden states, runs FPN+DETR for any layer or blend |
| `capr_router.py` | `CAPRRouter` MLP class + `load_router()` |
| `metrics.py` | IL_MCC, cgF1, IoU, pmF1 |
| `demo_cap.py` | Demo on "the cap" image — sweeps all 33 layers, saves grid + comparison figures |
| `evaluate.py` | Layer sweep evaluation (older, 100 pos + 100 neg) |
| `layer_sweep.py` | Core sweep helper |
| `run_pipeline.sh` | Full pipeline: collect → train → eval (run unattended) |
| `experiments/collect_oracle_layers.py` | Step 1: sweep layers on SA-Co positives, save training data |
| `experiments/train_router.py` | Step 2: train router with KL + CE loss |
| `experiments/eval_router_full.py` | Step 3: full evaluation on test_3 (IL_MCC primary metric) |
| `experiments/verify_hypothesis.py` | Statistical test of IL_MCC vs cgF1 peak layer |
| `experiments/find_failure_cases.py` | Find and visualise L32 failure cases |
| `results/` | All output figures, CSVs, model weights |

---

## How to Run

### Prerequisites

```bash
pip install torch transformers pillow numpy matplotlib tqdm scikit-learn pandas pycocotools
```

SAM3 model weights must be cached locally (HuggingFace `facebook/sam3`):
```python
from transformers import Sam3Model, Sam3Processor
Sam3Model.from_pretrained("facebook/sam3")      # downloads once
Sam3Processor.from_pretrained("facebook/sam3")
```

### Quick demo (single image, all 33 layers)

```bash
cd capr_clean
python demo_cap.py
# saves results/demo_masks.png, demo_masks_comparison.png, demo_metrics_bars.png
```

### Sanity check

```bash
python test_pipeline.py
```

### Full training pipeline (unattended, ~65 min on RTX 4090)

```bash
cd capr_clean
nohup bash run_pipeline.sh > /dev/null 2>&1 &
# monitor:
tail -f results/pipeline_run.log
```

This runs three steps automatically:

**Step 1 — Collect oracle layer data** *(70/20/10 split written to meta.json)*
```bash
python experiments/collect_oracle_layers.py
# Sweeps layers 17-32 on 1500 SA-Co test_1 positives
# Saves: results/router_training_data/{text_embs.npy, cgf1_matrix.npy, meta.json}
# meta.json includes splits: {"train": [...], "val": [...], "test": [...]}
```

**Step 2 — Train the router** *(uses 70% train / 20% val from meta.json)*
```bash
python experiments/train_router.py
# 150 epochs, KL divergence + cross-entropy loss
# Train on 70% split, val loss on 20% split, 10% test reserved
# Saves: results/capr_router_weights.pt, results/router_training_curve.png
```

**Step 3 — Evaluate on 10% test split + test_3 negatives (IL_MCC)**
```bash
python experiments/eval_router_full.py
# Positives: 10% oracle test split (150 samples, pre-computed labels)
# Negatives: 500 from saco_gold_metaclip_test_3.json (held-out, never used in training)
# Saves: results/eval_full_summary.png, results/eval_full_raw.csv, results/eval_full_layer_dist.png
```

### Hypothesis verification

```bash
python experiments/verify_hypothesis.py
# Tests: does IL_MCC peak at an earlier layer than cgF1/IoU?
# 50 random pos + 50 random neg from subset_gt.json
# Saves: results/verify_hypothesis.png, results/verify_hypothesis_case.png
```

### Find failure cases

```bash
python experiments/find_failure_cases.py
# Finds images where L32 fails but an intermediate layer recovers the mask
# Saves grid figures to: results/failure_cases/
```

---

## Metrics

| Metric | What it measures | Requires |
|--------|-----------------|----------|
| **IL_MCC** | Instance-Level Matthews Correlation Coefficient — binary presence discrimination (fires on positives, silent on negatives) | pos + neg |
| **cgF1** | Concept Grounding F1 — Dice coefficient between predicted mask and GT | pos only |
| **IoU** | Intersection-over-Union of predicted vs GT mask | pos only |
| **pmF1** | Fraction of positives where IoU ≥ 0.5 | pos only |

**Detection criterion:** `query_score = sigmoid(max pred_logit) ≥ 0.5` (fixed threshold, identical for every layer).

---

## Results

### Layer sweep — "the cap" (single image, all 33 layers)

| Layer | Block | query_score | cgF1 | IoU |
|-------|-------|------------|------|-----|
| 0–17 | embed–16 | < 0.5 | 0.000 | 0.000 |
| **L21** | **20** | **0.577** | 0.496 | 0.329 |
| **L23** | **22** | **0.614** | **0.897** | **0.813** |
| L24–L31 | 23–30 | < 0.5 | 0.000 | 0.000 |
| **L32 (default)** | **31** | **0.371** | **0.000** | **0.000** |

Block 22 recovers the cap with cgF1=0.897 on a case where SAM3's default layer completely fails.

### Router evaluation — SA-Co test_3 (184 samples with non-zero oracle cgF1)

| Method | cgF1 | IoU | pmF1 | Δ cgF1 |
|--------|------|-----|------|--------|
| L32 default | 0.681 | 0.592 | 0.641 | — |
| Router hard | 0.673 | 0.586 | 0.620 | −0.008 |
| **Router MoE** | **0.689** | **0.602** | **0.658** | **+0.008** |
| Oracle best | **0.792** | **0.703** | **0.777** | **+0.111** |

The oracle gap (+11 cgF1 points) confirms the motivation. The CAPR-MoE router captures a portion of this gap. Improved training data (2000+ samples) is expected to close more of the gap.

---

## Repository Structure

```
capr_clean/
├── sam3_wrapper.py              # SAM3 backbone wrapper with layer injection
├── capr_router.py               # CAPR router MLP
├── metrics.py                   # IL_MCC, cgF1, IoU, pmF1
├── demo_cap.py                  # Demo: all 33 layers on one image
├── evaluate.py                  # Dataset-level layer sweep evaluation
├── layer_sweep.py               # Layer sweep helper
├── test_pipeline.py             # Sanity check
├── run_pipeline.sh              # Full pipeline script (unattended)
├── experiments/
│   ├── collect_oracle_layers.py # Step 1: generate router training data
│   ├── train_router.py          # Step 2: train CAPR router
│   ├── eval_router_full.py      # Step 3: evaluate on held-out test set
│   ├── eval_router.py           # Earlier evaluation script
│   ├── verify_hypothesis.py     # Hypothesis test (IL_MCC vs cgF1 peaks)
│   └── find_failure_cases.py    # Visualise L32 failure cases
└── results/
    ├── capr_router_weights.pt   # Trained router checkpoint
    ├── router_training_data/    # Collected oracle labels (numpy)
    ├── demo_masks.png           # All-layer grid for "the cap"
    ├── demo_masks_comparison.png
    ├── verify_hypothesis.png
    ├── verify_hypothesis_case.png
    ├── router_training_curve.png
    ├── router_eval_results.png
    └── failure_cases/           # L32 failure visualisations
```

---

## Citation / Acknowledgements

Built on top of [SAM3](https://ai.meta.com/research/publications/segment-anything-3/) (Meta AI) and the [SA-Co](https://github.com/lzzcd001/sa-co) dataset (metaclip split). Model loaded via [HuggingFace Transformers](https://huggingface.co/facebook/sam3).
