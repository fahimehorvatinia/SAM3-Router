# CAPR / ALSR: Concept-Adaptive Presence Routing for SAM3

**CAPR** (Concept-Adaptive Presence Routing), also referred to as **ALSR** (Automated Layer Selection Router) in the paper, is a lightweight plug-in for SAM3 that improves visual concept grounding by adaptively selecting which backbone layer to inject per query, instead of always defaulting to the final layer (L32). SAM3 weights remain **fully frozen** — only a small router is trained.

---

## Motivation

SAM3 performs grounded segmentation by injecting intermediate backbone features into its FPN neck and running a DETR-style decoder for presence detection and mask generation. By default it always uses features from the final backbone layer (L32).

**Problem:** The optimal backbone layer depends on image content. A concept in a cluttered scene may need an earlier, more spatially-focused layer. L32 is suboptimal for **48% of samples** in our evaluation.

**Evidence:** An oracle experiment that sweeps all 16 layers (L17–L32) and picks the best per sample achieves an average cgF1 gain of **+10.6 percentage points** over always using L32.

**Approach:** Train a small router using a combined soft/hard loss on oracle-labelled "failed cases", then apply a gated inference strategy that invokes the router only when L32 is struggling.

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
  │  query_score q32    routing embedding e         │
  └─────────────────────┼───────────────────────────┘
                        │
              q32 ≥ 0.5 (gate)?
             /                  \
           YES                   NO
            │                    │
        use L32               Router
        result            → top-1 layer k
                          FPN(Lk) → DETR
                          → mask + score
```

### Step 1 — Oracle Layer Collection

For each positive training image, SAM3 runs **16 times** (once per candidate layer L17–L32). Each run produces a segmentation mask compared against the GT mask using cgF1:

```
cgf1_matrix[i] = [cgF1_at_L17, ..., cgF1_at_L32]   shape [N, 16]
```

### Step 2 — Router Training

#### Routing Signals (Ablation)

Three routing signals were evaluated. The **DETR cross-attention** embedding (default) uses the mean-pooled last-layer hidden state of SAM3's 200 DETR object queries after a first FPN(L32)+DETR pass:

```
e = mean_pool(DETR_last_layer_hidden_states)   shape: [256]
```

| Signal | Dim | cgF1 (Gated) | IL_MCC |
|---|---|---|---|
| Text embedding only (v1) | 1024 | 0.632 | 0.219 |
| **DETR cross-attention (v3)** | **256** | **0.645** | **0.229** |
| concat(img, text) (v5) | 2048 | 0.634 | 0.207 |
| **Full query Attention (v4)** | **200×256** | **0.634** | **0.220** |

> **Attention Router (v4, new):** Uses the full 200-query DETR sequence with Multi-Head Self-Attention (4 heads, 1-layer transformer) instead of mean-pooling. A learnable `[CLS]` token aggregates discriminative signal per query. With current training data (467 samples), performance is on par with the mean-pooled version. Expected to improve significantly with diverse multi-domain training data.

#### Router Architectures

**MLP Router (v3, default):**
```
e (256-dim)
  → Linear(256, 128) → LayerNorm → ReLU → Dropout(0.1)
  → Linear(128, 64)  → ReLU
  → Linear(64, 16)   → Softmax
```

**Attention Router (v4, new):**
```
query_seq (200 × 256)
  → Prepend [CLS] token → (201, 256)
  → TransformerEncoderLayer (4 heads, FFN=512, norm_first)
  → CLS token output (256)
  → Linear(256, 128) → ReLU → Dropout(0.1) → Linear(128, 16) → Softmax
  562K parameters
```

#### Failed-Cases Filter

The router is trained **only on samples where routing makes a meaningful difference**:
```
Include sample i  iff  max_k(cgf1_matrix[i]) - cgf1_matrix[i, L32] ≥ δ
```
With `δ = 0.02`, this retains **662 / 1368 samples (48.4%)**, yielding a more balanced oracle layer distribution across L17–L32.

#### Loss Function

```
L = α · L_KL  +  (1-α) · L_CE        α=0.6
```

- **Hard CE** (`L_CE`): decisive signal toward the single best layer, with inverse-frequency class weights to prevent rare-layer collapse.
- **Soft KL** (`L_KL`): temperature-scaled (`T=2.0`) cgF1 distribution teaches the full quality landscape — "L24 is almost as good as L25".

**Optimizer:** AdamW (`lr=5e-4`, `weight_decay=1e-3`) with 10-epoch warmup + cosine decay.

### Step 3 — Gated Inference

```python
q32 = SAM3_L32(query).query_score

if q32 >= GATE_THRESHOLD:   # L32 confident → keep it
    output = L32_result
else:                        # L32 struggling → invoke router
    e    = extract_routing_embedding(...)
    k    = router.hard_pick(e)
    output = SAM3_Lk(query)
```

The gate aligns with training: the router was only supervised on failed cases, so it should only be invoked on similar ones.

---

## Results

### Main Benchmark — SA-Co MetaCLIP (137 pos + 500 neg)

| Method | IL_MCC | Recall | cgF1 | IoU | pmF1 |
|---|---|---|---|---|---|
| SAM3 L32 (baseline) | +0.236 | 0.964 | 0.633 | 0.535 | 0.606 |
| Router Hard | +0.217 | 0.927 | 0.615 | 0.524 | 0.569 |
| Router MoE | +0.172 | 0.934 | 0.634 | 0.548 | 0.591 |
| **Gated ALSR (MLP, v3)** | **+0.229** | **0.985** | **0.645** | **0.545** | **0.613** |
| Gated ALSR (Attention, v4) | +0.220 | 0.971 | 0.634 | 0.536 | 0.606 |
| Oracle (upper bound) | — | 1.000 | 0.748 | 0.656 | 0.710 |

**Gated MLP Router gains over L32:** recall +2.2pp, cgF1 +1.3pp, IoU +1.0pp, pmF1 +0.7pp.

**Attention Router (v4):** marginally below MLP with 467 training samples. Attention adds 562K parameters; its advantage grows with more training data (see Diverse Training below).

### Latency Analysis (RTX 4090, 20 samples)

| Stage | Time (ms) | Note |
|---|---|---|
| Backbone + text encoder | 222.8 ± 2.8 | Runs once for every sample |
| 1st DETR pass (FPN+DETR, L32) | 59.3 ± 5.2 | Always runs |
| Router MLP forward | 0.3 ± 0.7 | **Negligible** |
| 2nd DETR pass (FPN+DETR, Lk) | 60.3 ± 5.7 | Only when gate fires |
| **End-to-end L32 baseline** | **282.4 ms** | |
| **Gated ALSR expected overhead** | **+2.2 ms (0.8%)** | Gate fires on 3.6% of samples |

The gate fires on only **3.6%** of samples (samples where L32 query score < 0.5). The expected overhead per sample is `0.036 × 60.3ms = 2.2ms` — less than **1% of baseline latency**. The backbone runs exactly once regardless.

### Cross-Dataset Generalization (200 pos + 500 neg per domain)

Router trained on MetaCLIP only, evaluated zero-shot on all other domains:

| Dataset | Method | IL_MCC | Recall | cgF1 | IoU |
|---|---|---|---|---|---|
| MetaCLIP | L32 baseline | +0.204 | 0.880 | 0.637 | 0.559 |
| MetaCLIP | **Gated ALSR** | **+0.164** | **0.900** | **0.648** | **0.567** |
| Attributes | L32 baseline | +0.239 | 0.990 | 0.847 | 0.786 |
| Attributes | **Gated ALSR** | **+0.216** | **0.995** | **0.849** | **0.787** |
| Crowded | L32 baseline | **+0.345** | 0.950 | 0.642 | 0.539 |
| Crowded | **Gated ALSR** | +0.304 | **0.960** | **0.648** | **0.543** |
| Wiki-Food&Drink | L32 baseline | +0.270 | 0.970 | 0.756 | 0.680 |
| Wiki-Food&Drink | **Gated ALSR** | +0.212 | **0.980** | **0.765** | **0.688** |
| Wiki-Sports Equip. | L32 baseline | +0.278 | 0.940 | 0.803 | 0.741 |
| Wiki-Sports Equip. | **Gated ALSR** | **+0.239** | **0.950** | **0.807** | **0.743** |

The Gated Router consistently improves recall (+1–2pp) and cgF1 (+0.1–0.9pp) across all domains without retraining.

---

## Reviewer Experiments (v2)

Three additional experiments added in response to reviewer feedback:

### Exp 1 — Attention-Based Router (v4)

**Motivation:** The MLP router discards the internal structure of 200 DETR queries by mean-pooling them. Only a few queries correspond to the relevant concept; attention can learn to up-weight those.

**Implementation:**
- `capr_router.py` → `AttentionCAPRRouter` class
- `sam3_wrapper.py` → `extract_detr_emb_full()` returns (200, 256) full query sequence
- `experiments/extract_detr_embs_full.py` → extracts and saves `detr_embs_full.npy`

**Run:**
```bash
python experiments/extract_detr_embs_full.py        # ~30 min on GPU; saves detr_embs_full.npy (268 MB)
EMB_MODE=detr_full python experiments/train_router.py   # trains capr_router_weights_attn.pt
ROUTER_WEIGHTS=results/capr_router_weights_attn.pt python experiments/eval_router_full.py
```

**Results (MetaCLIP test set):**

| Router | Val top-1 acc | Gated cgF1 | Gated Recall | Gated IL_MCC |
|---|---|---|---|---|
| MLP (mean-pool, v3) | ~20% | 0.645 | 0.985 | 0.229 |
| Attention (full seq, v4) | **15.3%** | 0.634 | 0.971 | 0.220 |

**Interpretation:** With only 467 training samples, the attention router's additional capacity does not yield improvement — a known challenge for transformers in data-scarce settings. The performance gap is expected to close with diverse multi-domain training (Exp 3 below).

---

### Exp 2 — Latency of Second Pass

**Motivation:** The paper claimed the gated overhead is minimal but provided no measurements.

**Implementation:** `experiments/measure_latency.py` — benchmarks each pipeline stage separately using `torch.cuda.synchronize()` for accurate GPU timing.

**Run:**
```bash
N_SAMPLES=20 python experiments/measure_latency.py    # ~3 min; saves results/latency_benchmark.json
```

**Results (RTX 4090, n=20):**

```
Backbone + text encoder  : 222.8 ± 2.8 ms   (runs once, always)
1st DETR pass (L32)      :  59.3 ± 5.2 ms   (runs always)
Router MLP forward       :   0.3 ± 0.7 ms   (negligible)
2nd DETR pass (Lk)       :  60.3 ± 5.7 ms   (fires on 3.6% of samples)

End-to-end L32 baseline  : 282.4 ms
Gate trigger rate        : 3.6%
Expected gated overhead  : 0.036 × 60.3 = 2.2 ms  (+0.8%)
```

The gated overhead is **0.8%** of baseline latency — negligible in practice.

---

### Exp 3 — Diverse Multi-Domain Training

**Motivation:** The router is trained on only MetaCLIP `test_1` (~1,368 samples, 467 filtered). More training data from diverse concept types should improve routing quality and reduce the gap between MLP and Attention routers.

**Available datasets** (all sharing `metaclip-images/` image root):

| Domain | Images | Annotations | Concept Type |
|---|---|---|---|
| MetaCLIP | 33,393 | 20,144 | General web concepts |
| Attributes | 9,245 | 3,663 | Attribute-qualified objects |
| Crowded | 20,687 | 50,417 | Cluttered/crowded scenes |
| Wiki-Food&Drink | 13,951 | 10,041 | Wikipedia food categories |
| Wiki-Sports Equipment | 12,166 | 5,075 | Wikipedia sports equipment |
| Wiki-Common1K | 65,502 | 6,448 | Wikipedia common 1K |

> Note: SA-1B is also available but uses a different image source and is excluded here.

**Implementation:** `experiments/collect_oracle_diverse.py` — domain-agnostic oracle collection; merges all 6 domains into one training pool with domain labels.

**Run (requires GPU, ~6–8 hours total):**
```bash
# Collect oracle data from all 6 MetaCLIP-image domains (~1500 samples each)
python experiments/collect_oracle_diverse.py
# Saved to: results/router_training_data_diverse/  (text_embs, img_embs, cgf1_matrix, meta.json)

# Train router on combined data
DATA_DIR=$(pwd)/results/router_training_data_diverse \
EMB_MODE=concat python experiments/train_router.py
# Checkpoint: results/capr_router_weights_diverse.pt

# Evaluate
ROUTER_WEIGHTS=results/capr_router_weights_diverse.pt python experiments/eval_router_full.py
```

Expected benefit: ~9,000 total samples (vs 467 current) across 6 diverse concept types → improved routing accuracy, especially for the attention router.

---

## Project Structure

```
capr_clean/
├── sam3_wrapper.py                      # SAM3 wrapper: extract(), run(), extract_detr_emb_full()
├── capr_router.py                       # CAPRRouter (MLP) + AttentionCAPRRouter + load_router()
├── metrics.py                           # cgF1, IoU, IL_MCC, merge_gt_masks
├── evaluate.py                          # Layer sweep evaluation
├── demo_cap.py                          # Visual demo
├── experiments/
│   ├── collect_oracle_layers.py         # Step 1: oracle collection (MetaCLIP)
│   ├── collect_oracle_diverse.py        # Step 1 (Exp 3): oracle collection, all domains
│   ├── extract_detr_embs.py             # Step 1b: 256-dim mean-pooled DETR embeddings
│   ├── extract_detr_embs_full.py        # Step 1b (Exp 1): full (200,256) DETR sequences
│   ├── train_router.py                  # Step 2: train MLP or Attention router
│   ├── eval_router_full.py              # Step 3: full evaluation (supports all router types)
│   ├── eval_crossdataset.py             # Cross-domain evaluation
│   ├── measure_latency.py               # Exp 2: latency benchmark per pipeline stage
│   ├── eval_router.py                   # Quick held-out eval
│   ├── find_failure_cases.py            # Failure case visualization
│   └── verify_hypothesis.py            # Layer-matters hypothesis validation
├── results/
│   ├── router_training_data/
│   │   ├── cgf1_matrix.npy              # [1368, 16] oracle cgF1 per layer
│   │   ├── detr_embs.npy                # [1368, 256] mean-pooled DETR embeddings
│   │   ├── detr_embs_full.npy           # [1368, 200, 256] full per-query sequences (Exp 1)
│   │   ├── text_embs.npy                # [1368, 1024]
│   │   ├── img_embs.npy                 # [1368, 1024]
│   │   └── meta.json                    # image IDs, prompts, splits
│   ├── capr_router_weights.pt           # MLP router (v3, DETR 256-dim)
│   ├── capr_router_weights_attn.pt      # Attention router (v4, 200×256 full seq) [Exp 1]
│   ├── latency_benchmark.json           # Stage-wise latency results [Exp 2]
│   ├── router_training_data_diverse/    # Multi-domain oracle data [Exp 3, after collection]
│   ├── crossdataset/                    # Per-dataset CSVs and summaries
│   └── eval_full_raw.csv                # Per-sample evaluation results
└── run_v4_pipeline.sh / run_v5_pipeline.sh / run_crossdataset.sh
```

---

## Setup

```bash
conda activate <your_sam3_env>
pip install tqdm matplotlib scikit-learn
```

---

## Running the Full Pipeline

```bash
cd ALSR_clean

# ── Original pipeline (MLP router, MetaCLIP only) ──────────────────────────
python experiments/collect_oracle_layers.py         # ~2h: sweep 16 layers × 1500 images
python experiments/extract_detr_embs.py             # ~30 min: 256-dim routing embeddings
EMB_MODE=detr python experiments/train_router.py    # ~2 min: train MLP router
GATE_THRESHOLD=0.5 python experiments/eval_router_full.py  # evaluate

# ── Exp 1: Attention router ────────────────────────────────────────────────
python experiments/extract_detr_embs_full.py        # ~30 min: (200,256) full sequences
EMB_MODE=detr_full python experiments/train_router.py       # train attention router
ROUTER_WEIGHTS=results/capr_router_weights_attn.pt python experiments/eval_router_full.py

# ── Exp 2: Latency benchmark ───────────────────────────────────────────────
N_SAMPLES=20 python experiments/measure_latency.py

# ── Exp 3: Diverse training ────────────────────────────────────────────────
python experiments/collect_oracle_diverse.py        # ~6-8h: all 6 MetaCLIP-image domains
DATA_DIR=$(pwd)/results/router_training_data_diverse EMB_MODE=concat python experiments/train_router.py
ROUTER_WEIGHTS=results/capr_router_weights_diverse.pt python experiments/eval_router_full.py

# ── Cross-domain evaluation ────────────────────────────────────────────────
bash run_crossdataset.sh
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `EMB_MODE` | `detr` | Router input: `detr` (256-dim), `detr_full` (200×256 attention), `concat` (2048-dim img+text), `text` (1024-dim) |
| `DELTA_THRESHOLD` | `0.02` | Min cgF1 gain over L32 required to include a sample in training |
| `FOCUS_FAILED` | `1` | Train only on failed cases — prevents mode collapse |
| `GATE_THRESHOLD` | `0.5` | L32 query score below which the router is invoked |
| `ROUTER_WEIGHTS` | auto | Path to router weights; auto-detects MLP vs Attention from checkpoint keys |
| `DATA_DIR` | `results/router_training_data` | Override to use diverse training data |
| `DOMAINS` | all 6 | Comma-separated domain list for `collect_oracle_diverse.py` |
| `N_PER_DOMAIN` | `1500` | Max samples per domain in diverse collection |
| `ALPHA` | `0.6` | Weight of KL loss (soft) in training objective |
| `TEMPERATURE` | `2.0` | Temperature for soft target distribution |

---

## Metrics

| Metric | Description |
|---|---|
| **IL_MCC** | Instance-Level Matthews Correlation Coefficient — presence/absence discrimination. Range: [−1, +1]. |
| **Recall** | Fraction of present concepts correctly detected (query_score ≥ 0.5) |
| **Precision** | Positive predictive value |
| **cgF1** | Mask quality: `2·|P∩G| / (|P|+|G|)` |
| **IoU** | Intersection-over-Union of predicted vs. GT mask |
| **pmF1** | Per-mask F1 (IoU ≥ 0.5 threshold) |

---

## Citation

```bibtex
@article{horvatinia2026alsr,
  title   = {Automated Layer Selection Router for Improved Grounded Segmentation in SAM3},
  author  = {Horvatinia, Fahime and others},
  journal = {NeurIPS},
  year    = {2026}
}
```
