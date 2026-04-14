# CAPR: Concept-Adaptive Presence Routing for SAM3

**CAPR** (Concept-Adaptive Presence Routing) is a lightweight plug-in for [SAM3](https://github.com/...) that improves its visual concept grounding performance on difficult cases by adaptively selecting which backbone layer to use for each query, rather than always defaulting to the final layer (L32).

---

## Motivation

SAM3 performs grounded segmentation by injecting intermediate backbone features into its FPN neck and running a DETR-style decoder for presence detection and mask generation. By default it always uses features from the final backbone layer (L32).

**Problem:** The optimal backbone layer depends on image content — a concept in a cluttered scene may need an earlier, more spatially-focused layer. L32 is suboptimal for ~48% of samples in our evaluation.

**Evidence:** An oracle experiment that sweeps all 16 layers (L17–L32) and picks the best per sample achieves an average cgF1 gain of **+10.6 percentage points** over always using L32. This gap represents recoverable performance.

**Approach:** Train a small MLP router that predicts which backbone layer to use given the current image-text query, and apply a gated inference strategy that invokes the router only when L32 is struggling.

---

## Method

### Architecture

```
Image + Text Query
       │
  SAM3 Backbone (frozen)
       │
  hidden_states[17..32]           ← 16 candidate layers
       │
  ┌────┴────────────────────────────┐
  │  Routing Path (v4c)             │
  │  FPN(L32) → DETR → 256-dim emb │
  │        ↓                        │
  │  CAPR Router MLP (256→128→64→16)│
  │        ↓ top-1 layer k          │
  └────────┼────────────────────────┘
           │
  FPN(Lk) → DETR → mask + query_score
```

### Oracle Layer Collection (Step 1)

For each training image, run SAM3 with all 16 layers and record the cgF1 (complementary global F1) of the resulting mask at each layer. The layer with highest cgF1 is the oracle label.

### Router Training (Step 2)

- **Input**: DETR cross-attention embedding (256-dim) from the SAM3 decoder
- **Loss**: α·KL-divergence (soft targets from cgF1 distribution) + (1-α)·CrossEntropy (hard oracle label)
- **Filtering**: Train only on "failed cases" where `max_cgF1(all layers) − cgF1(L32) ≥ 0.05` — this prevents mode collapse toward always predicting L32
- **Class weights**: Inverse-frequency weighting to handle layer distribution imbalance

### Gated Inference (Step 3)

```
q32 = SAM3_L32(query).query_score

if q32 ≥ GATE_THRESHOLD:          # L32 confident → use it
    output = SAM3_L32_result
else:                              # L32 struggling → ask the router
    k = router.hard_pick(emb)
    output = SAM3_Lk(query)
```

This aligns with training: the router was only trained on failed cases, so it should only be invoked on similar ones.

---

## Results

Evaluated on SA-Co MetaCLIP test split: **137 positives + 500 negatives**.
Primary metric: **IL_MCC** (Instance-Level Matthews Correlation Coefficient for presence detection).

| Method | IL_MCC | Recall | cgF1 | IoU | pmF1 |
|---|---|---|---|---|---|
| SAM3 L32 (baseline) | +0.2358 | 0.9635 | 0.6325 | 0.5352 | 0.6058 |
| Router Hard (CAPR top-1) | +0.2165 | 0.9270 | 0.6150 | 0.5237 | 0.5693 |
| Router MoE (CAPR soft blend) | +0.1721 | 0.9343 | 0.6341 | 0.5479 | 0.5912 |
| **Gated CAPR Router** | **+0.2286** | **0.9854** | **0.6453** | **0.5448** | **0.6131** |
| Oracle (upper bound) | — | 1.0000 | 0.7481 | 0.6564 | 0.7100 |

**Key findings:**
- The Oracle ceiling (+11.6pp cgF1 over L32) confirms layer selection matters significantly
- Blind routing (Router Hard/MoE) hurts because it overrides confident L32 predictions
- The Gated Router is the best strategy: recall +2.2pp, cgF1 +1.3pp, IoU +1.0pp, pmF1 +0.7pp
- There remains a 10pp cgF1 gap to the oracle — a better routing signal is the open challenge

---

## Project Structure

```
capr_clean/
├── sam3_wrapper.py              # SAM3 wrapper: extract embeddings, inject layers, run DETR
├── capr_router.py               # CAPRRouter MLP definition + load_router()
├── metrics.py                   # cgF1, IoU, IL_MCC, merge_gt_masks
├── evaluate.py                  # Full evaluation entry point
├── demo_cap.py                  # Visual demo: side-by-side mask comparison
├── experiments/
│   ├── collect_oracle_layers.py # Step 1: sweep L17-L32, record cgF1 per layer
│   ├── extract_detr_embs.py     # Step 1b: extract DETR routing embeddings
│   ├── train_router.py          # Step 2: train CAPR router MLP
│   ├── eval_router_full.py      # Step 3: full evaluation (IL_MCC, cgF1, IoU, pmF1)
│   ├── eval_router.py           # Quick evaluation (detection metrics only)
│   ├── find_failure_cases.py    # Identify and visualize failed cases
│   └── verify_hypothesis.py     # Validate layer-matters hypothesis
├── results/
│   ├── router_training_data/    # oracle cgF1 matrix, embeddings, splits
│   │   ├── cgf1_matrix.npy      # [1368, 16] oracle cgF1 per layer
│   │   ├── detr_embs.npy        # [1368, 256] DETR routing embeddings
│   │   ├── text_embs.npy        # [1368, 1024] text embeddings
│   │   ├── img_embs.npy         # [1368, 1024] image embeddings
│   │   └── meta.json            # image IDs, prompts, train/val/test splits
│   ├── capr_router_weights.pt   # trained router checkpoint
│   └── eval_full_raw.csv        # per-sample evaluation results
├── run_v4_pipeline.sh           # end-to-end pipeline (best config)
└── sam3/                        # SAM3 model code (submodule)
```

---

## Setup

```bash
# 1. Install dependencies (same environment as SAM3)
conda activate <your_sam3_env>
pip install tqdm matplotlib scikit-learn

# 2. Set data paths in collect_oracle_layers.py and eval_router_full.py:
DATA_FILE  = "/path/to/saco_gold_data/metaclip/saco_gold_metaclip_test_1.json"
IMAGE_ROOT = "/path/to/metaclip-images"
```

---

## Running the Full Pipeline

```bash
cd capr_clean

# Step 1: Collect oracle layer data (runs SAM3 × 16 layers per image, ~2h for 1500 images)
python experiments/collect_oracle_layers.py

# Step 1b: Extract DETR routing embeddings
python experiments/extract_detr_embs.py

# Step 2: Train router (< 2 min on GPU)
EMB_MODE=detr DELTA_THRESHOLD=0.05 python experiments/train_router.py

# Step 3: Evaluate
GATE_THRESHOLD=0.5 python experiments/eval_router_full.py
```

Or run the complete pipeline in one command:
```bash
nohup bash run_v4_pipeline.sh > results/pipeline_v4.log 2>&1 &
```

---

## Key Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `EMB_MODE` | `detr` | Router input: `detr` (256-dim), `concat` (2048-dim img+text), `text` (1024-dim) |
| `DELTA_THRESHOLD` | `0.05` | Min cgF1 gain over L32 for a sample to be used in training |
| `FOCUS_FAILED` | `1` | Train only on failed cases (1=yes, prevents mode collapse) |
| `GATE_THRESHOLD` | `0.5` | L32 query score below which the gated router is invoked |
| `ALPHA` | `0.6` | KL loss weight (vs CE) in router training |
| `EPOCHS` | `150` | Training epochs for router MLP |

---

## Metrics

| Metric | Description |
|---|---|
| **IL_MCC** | Instance-Level Matthews Correlation Coefficient — measures presence/absence discrimination quality |
| **Recall** | True positive rate for presence detection (query_score ≥ 0.5) |
| **Precision** | Positive predictive value |
| **cgF1** | Complementary global F1 — mask quality metric: harmonic mean of mask precision and recall |
| **IoU** | Intersection-over-Union of predicted vs ground-truth mask |
| **pmF1** | Per-mask F1 score |

---

## Citation

```bibtex
@article{horvatinia2026capr,
  title   = {CAPR: Concept-Adaptive Presence Routing for SAM3},
  author  = {Horvatinia, Fahime},
  year    = {2026}
}
```
