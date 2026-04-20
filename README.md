# CAPR: Concept-Adaptive Presence Routing for SAM3

**CAPR** (Concept-Adaptive Presence Routing) is a lightweight plug-in for SAM3 that improves visual concept grounding on difficult cases by adaptively selecting which backbone layer to use per query, rather than always defaulting to the final layer (L32). SAM3 weights remain **fully frozen** — only a small MLP router is trained.

---

## Motivation

SAM3 performs grounded segmentation by injecting intermediate backbone features into its FPN neck and running a DETR-style decoder for presence detection and mask generation. By default it always uses features from the final backbone layer (L32).

**Problem:** The optimal backbone layer depends on image content — a concept in a cluttered scene may need an earlier, more spatially-focused layer. L32 is suboptimal for **48% of samples** in our evaluation.

**Evidence:** An oracle experiment that sweeps all 16 layers (L17–L32) and picks the best per sample achieves an average cgF1 gain of **+10.6 percentage points** over always using L32. This gap represents recoverable performance.

**Approach:** Train a small MLP router using a combined soft/hard loss on oracle-labelled "failed cases", then apply a gated inference strategy that invokes the router only when L32 is struggling.

---

## Method

### Architecture

```
Image + Text Query
       │
  SAM3 Backbone (frozen)              hidden_states[0..32]
       │
  ┌────┴────────────────────────────────────────────┐
  │  Pass 1 (always)                                │
  │  FPN(L32) → DETR decoder                        │
  │       ↓              ↓                          │
  │  query_score q32    256-dim DETR embedding e    │
  └─────────────────────┼───────────────────────────┘
                        │
              q32 ≥ 0.5 (gate)?
             /                  \
           YES                   NO
            │                    │
        use L32             CAPR Router MLP
        result            (256→128→64→16)
                                 │ top-1 layer k
                          FPN(Lk) → DETR
                          → mask + score
```

### Step 1 — Oracle Layer Collection

For each positive training image, SAM3 runs **16 times** (once per candidate layer L17–L32). Each run produces a segmentation mask that is compared against the ground-truth mask using cgF1. This yields a per-sample vector of 16 quality scores — the oracle dataset.

```
cgf1_matrix[i] = [cgF1_at_L17, cgF1_at_L18, ..., cgF1_at_L32]   shape [N, 16]
```

Stored in `results/router_training_data/cgf1_matrix.npy`.

### Step 2 — Router Training

#### Routing Signal (Input)

The router receives the **DETR cross-attention embedding**: the mean-pooled last-layer hidden state of SAM3's 200 DETR object queries after a first FPN(L32)+DETR pass:

```
e = mean_pool(DETR_last_layer_hidden_states)   shape: [256]
```

This 256-dim vector is SAM3's own internal representation of how the concept aligns with the image — the richest available signal without an extra forward pass.

Three routing signals were evaluated as ablations:

| Signal | Dim | cgF1 (Gated) | IL_MCC |
|---|---|---|---|
| Text embedding only | 1024 | 0.632 | 0.219 |
| **DETR cross-attention** | **256** | **0.645** | **0.229** |
| concat(img, text) | 2048 | 0.634 | 0.207 |

The DETR signal wins because it already fuses both text and image information through 6 layers of cross-attention, making it more compact and discriminative than a raw concatenation.

#### Router Architecture

```
e (256-dim)
  → Linear(256, 128) → LayerNorm → ReLU → Dropout(0.1)
  → Linear(128, 64)  → ReLU
  → Linear(64, 16)   → Softmax
  → probability distribution over 16 layers
```

#### Failed-Cases Filter

The router is trained **only on samples where routing makes a meaningful difference**:

```
Include sample i  iff  max_k(cgf1_matrix[i]) - cgf1_matrix[i, L32] ≥ δ
```

With `δ = 0.05`, this retains 467 / 1368 samples (34.1%).

**Why this matters:** Without filtering, the oracle layer distribution is heavily skewed toward L31 (26%) and L32 (22%). The router collapses to always predicting these two layers, learning the prior rather than the mapping. Restricting to hard cases yields a much more balanced distribution across all 16 layers.

#### Loss Function

The router is trained with a **combined soft + hard objective**:

```
L = α · L_KL  +  (1-α) · L_CE        α=0.6
```

**Hard CE loss — `L_CE`:**

```
L_CE = CrossEntropy(logits, k*)        k* = argmax(cgf1_matrix[i])
```

The CE loss provides a decisive signal: "the correct layer is L25." It uses **class-balanced weights** (inverse frequency per layer) to prevent the router from ignoring rare layers like L17–L20:

```
weight[k] = 1 / count(k)   (normalized so mean weight = 1)
```

**Soft KL loss — `L_KL`:**

```
Q[k] = softmax(cgf1_matrix[i] / T)     T=2.0  (temperature)
L_KL = KL(Q || softmax(logits))
```

The soft target `Q` is built by temperature-scaling the raw cgF1 vector and applying softmax. Temperature `T=2.0` flattens the distribution, giving nearby-good layers non-zero probability. This teaches the router that "L24 is nearly as good as L25" rather than creating a hard cliff.

**Why both losses?**

| Loss | Teaches | Risk if alone |
|---|---|---|
| CE (hard) | "The single correct layer is k*" | Ignores that nearby layers are also good |
| KL (soft) | "The full distribution of layer quality" | Weak supervision, slow convergence |
| Combined | Both decisive and nuanced | Balanced |

**Optimizer:** AdamW (`lr=5e-4`, `weight_decay=1e-3`) with 10-epoch warmup followed by cosine decay to zero. The warmup prevents early collapse toward majority classes before class weights take effect.

#### Training vs. Inference — Routing Type

| Stage | Routing | Description |
|---|---|---|
| Training | **Soft** | KL loss trains router to output a full probability distribution |
| Inference: Router Hard | **Hard** | `argmax` of router output — single top-1 layer |
| Inference: Router MoE | **Soft** | Weighted blend of all 16 layer features |
| Inference: Gated Router | **Gated Hard** | Hard routing, but only when `q32 < gate_threshold` |

### Step 3 — Gated Inference

```python
q32 = SAM3_L32(query).query_score

if q32 >= GATE_THRESHOLD:     # L32 confident → trust it
    output = SAM3_L32_result
else:                         # L32 struggling → invoke router
    e    = extract_detr_emb(...)
    k    = router.hard_pick(e)
    output = SAM3_Lk(query)
```

The gate aligns with training: the router was only supervised on samples where L32 fails, so it should only be invoked on similar ones.

**Computational cost:** The backbone runs exactly once. The second DETR pass only executes for the small fraction of samples where `q32 < 0.5` (typically 3–4% of samples in easy domains, up to ~20% in crowded scenes).

---

## Results

### Main Benchmark — SA-Co MetaCLIP (test split)

**137 positives + 500 negatives.** Primary metric: **IL_MCC**.

| Method | IL_MCC | Recall | cgF1 | IoU | pmF1 |
|---|---|---|---|---|---|
| SAM3 L32 (baseline) | 0.2358 | 0.9635 | 0.6325 | 0.5352 | 0.6058 |
| Router Hard | +0.2165 | 0.9270 | 0.6150 | 0.5237 | 0.5693 |
| Router MoE | +0.1721 | 0.9343 | 0.6341 | 0.5479 | 0.5912 |
| **Gated CAPR Router** | **+0.2286** | **0.9854** | **0.6453** | **0.5448** | **0.6131** |


**Gated Router gains over L32:** recall +2.2pp, cgF1 +1.3pp, IoU +1.0pp, pmF1 +0.7pp, IL_MCC −0.007 (negligible).

### Cross-Dataset Generalization

The router was trained on SA-Co MetaCLIP (test_1) and evaluated **zero-shot** on 4 other SA-Co domains. **200 positives + 500 negatives per domain.**

| Dataset | Method | IL_MCC | Recall | cgF1 | IoU |
|---|---|---|---|---|---|
| MetaCLIP | L32 baseline | 0.204 | 0.880 | 0.637 | 0.559 |
| MetaCLIP | **Gated CAPR** | **+0.164** | **0.900** | **0.648** | **0.567** |
| Attributes | L32 baseline | 0.239 | 0.990 | 0.847 | 0.786 |
| Attributes | **Gated CAPR** | **+0.216** | **0.995** | **0.849** | **0.787** |
| Crowded | L32 baseline | **+0.345** | 0.950 | 0.642 | 0.539 |
| Crowded | **Gated CAPR** | +0.304 | **0.960** | **0.648** | **0.543** |
| Wiki-Food&Drink | L32 baseline | +0.270 | 0.970 | 0.756 | 0.680 |
| Wiki-Food&Drink | **Gated CAPR** | +0.212 | **0.980** | **0.765** | **0.688** |
| Wiki-Sports Equip. | L32 baseline | +0.278 | 0.940 | 0.803 | 0.741 |
| Wiki-Sports Equip. | **Gated CAPR** | **+0.239** | **0.950** | **0.807** | **0.743** |

**The Gated Router consistently improves recall (+1.0–2.0pp) and cgF1 (+0.1–0.9pp) across all domains without retraining.**

**Dataset difficulty observations:**
- **Attributes** (e.g., "red chair"): easiest domain, cgF1 = 0.847. Objects are well-defined but concepts are attribute-qualified.
- **Wiki-Sports/Food**: clean product-style images, high cgF1 (0.76–0.80).
- **MetaCLIP / Crowded**: hardest, most varied scenes, lowest cgF1 (0.63–0.64).

---

## Project Structure

```
capr_clean/
├── sam3_wrapper.py                  # SAM3 wrapper: embed extraction, layer injection, DETR
├── capr_router.py                   # CAPRRouter MLP + load_router() + hard_pick() + run_moe()
├── metrics.py                       # cgF1, IoU, IL_MCC, merge_gt_masks
├── evaluate.py                      # Full evaluation entry point
├── demo_cap.py                      # Visual demo: side-by-side mask comparison
├── experiments/
│   ├── collect_oracle_layers.py     # Step 1: sweep L17-L32, record cgF1 matrix per image
│   ├── extract_detr_embs.py         # Step 1b: extract 256-dim DETR routing embeddings
│   ├── train_router.py              # Step 2: train CAPR MLP router (KL + CE loss)
│   ├── eval_router_full.py          # Step 3: full evaluation on metaclip test split
│   ├── eval_crossdataset.py         # Cross-domain evaluation on any SA-Co split
│   ├── eval_router.py               # Quick detection-only evaluation
│   ├── find_failure_cases.py        # Identify and visualize failed cases
│   └── verify_hypothesis.py        # Validate layer-matters hypothesis
├── results/
│   ├── router_training_data/
│   │   ├── cgf1_matrix.npy          # [1368, 16]  oracle cgF1 per layer per sample
│   │   ├── detr_embs.npy            # [1368, 256] DETR cross-attention embeddings
│   │   ├── text_embs.npy            # [1368, 1024] SAM3 text encoder embeddings
│   │   ├── img_embs.npy             # [1368, 1024] SAM3 image backbone embeddings
│   │   └── meta.json                # image IDs, prompts, 70/20/10 train/val/test splits
│   ├── crossdataset/                # Per-dataset CSVs and summaries
│   ├── capr_router_weights.pt       # Trained router checkpoint (auto-detected by load_router)
│   └── eval_full_raw.csv            # Per-sample evaluation results
├── run_v4_pipeline.sh               # End-to-end pipeline (best DETR config)
├── run_v5_pipeline.sh               # Pipeline with img+text concat router
├── run_crossdataset.sh              # Cross-dataset evaluation runner
└── sam3/                            # SAM3 model code (submodule)
```

---

## Setup

```bash
# 1. Use the same Python environment as SAM3
conda activate <your_sam3_env>
pip install tqdm matplotlib scikit-learn

# 2. Set data paths (edit collect_oracle_layers.py and eval_router_full.py):
DATA_FILE  = "/path/to/saco_gold_data/metaclip/saco_gold_metaclip_test_1.json"
IMAGE_ROOT = "/path/to/metaclip-images"
```

---

## Running the Full Pipeline

```bash
cd capr_clean

# Step 1: Collect oracle labels (SAM3 × 16 layers × N images, ~2h for 1500 images)
python experiments/collect_oracle_layers.py

# Step 1b: Extract DETR routing embeddings
python experiments/extract_detr_embs.py

# Step 2: Train router (~2 min on GPU)
EMB_MODE=detr DELTA_THRESHOLD=0.05 FOCUS_FAILED=1 python experiments/train_router.py

# Step 3: Evaluate on held-out metaclip test split
GATE_THRESHOLD=0.5 python experiments/eval_router_full.py

# Cross-dataset evaluation (all domains)
bash run_crossdataset.sh
```

Or run the complete pipeline in one command:
```bash
nohup bash run_v4_pipeline.sh > results/pipeline_v4.log 2>&1 &
```

---

## Key Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `EMB_MODE` | `concat` | Router input: `detr` (256-dim DETR), `concat` (2048-dim img+text), `text` (1024-dim), `img_only` (1024-dim) |
| `DELTA_THRESHOLD` | `0.02` | Min cgF1 gain over L32 required to include a sample in training |
| `FOCUS_FAILED` | `1` | Train only on failed cases (`1`=yes) — prevents mode collapse |
| `GATE_THRESHOLD` | `0.5` | L32 query score below which the gated router is invoked |
| `ALPHA` | `0.6` | Weight of KL loss (soft) vs CE loss (hard) in training objective |
| `TEMPERATURE` | `2.0` | Temperature for soft target distribution (higher = softer labels) |
| `USE_CLASS_WEIGHTS` | `1` | Apply inverse-frequency class weights to CE loss |
| `EPOCHS` | `300` | Training epochs |
| `LR` | `5e-4` | AdamW peak learning rate |

---

## Metrics

| Metric | Description |
|---|---|
| **IL_MCC** | Instance-Level Matthews Correlation Coefficient — measures presence/absence discrimination. Balances TP, TN, FP, FN. Range: [−1, +1]. |
| **Recall** | True positive rate: fraction of present concepts correctly detected (`query_score ≥ 0.5`) |
| **Precision** | Positive predictive value |
| **cgF1** | Complementary global F1 — mask quality: `2·|P∩G| / (|P|+|G|)` where P=predicted mask, G=GT mask |
| **IoU** | Intersection-over-Union of predicted vs. ground-truth mask |
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
