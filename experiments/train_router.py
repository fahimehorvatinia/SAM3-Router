"""
Step 2 of 3: Train the CAPR router on collected oracle layer data.

Loss = α × KL_divergence(router_output || soft_target)
     + (1-α) × CrossEntropy(router_output, hard_best_layer)

  soft_target: normalize cgF1 scores per sample with temperature T.
               This teaches the router the full distribution of layer quality,
               not just the top-1 label.

  hard_best_layer: argmax of cgF1 per sample (one-hot supervision).

Uses 80/20 train/val split. Saves best-val-loss checkpoint to
  results/capr_router_weights.pt

Usage:
    cd capr_clean
    python experiments/train_router.py
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capr_router import CAPRRouter

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "router_training_data")
OUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results")

EPOCHS     = 300
LR         = 5e-4
BATCH_SIZE = 64
# Split indices come from meta.json (70/20/10 created by collect_oracle_layers.py).
# Training uses the 70% train split; val uses the 20% val split.
# The 10% test split is reserved for eval_router_full.py.
ALPHA      = 0.6        # weight for soft KL loss (vs hard CE)
TEMPERATURE= 2.0        # softmax temperature for soft labels
SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Class-balanced CE: upweight rare layers (early layers under-represented even
# in the failed-cases subset).  Applied to the hard-CE term only.
USE_CLASS_WEIGHTS = os.environ.get("USE_CLASS_WEIGHTS", "1").lower() not in ("0","false","no")

# ── Failed-cases filter ────────────────────────────────────────────────────────
# Core design decision: we only want to IMPROVE cases where the default SAM3
# layer (L32) already fails.  The default SAM3 is already good on easy samples
# — training the router on those just teaches it "always predict L31/L32",
# leading to mode collapse.
#
# FOCUS_FAILED=True filters training to samples where routing can make a
# measurable difference:  max_cgF1_over_layers - cgF1_at_L32 >= DELTA_THRESHOLD
#
# DELTA_THRESHOLD=0.05 means "keep only samples where the best available layer
# is at least 5 percentage points better than L32 in mask quality (cgF1)".
# These are the samples where routing matters — the router is only supervised
# on meaningful signal, not on noise from already-solved cases.
DELTA_THRESHOLD = float(os.environ.get("DELTA_THRESHOLD", "0.02"))
FOCUS_FAILED    = os.environ.get("FOCUS_FAILED", "1").lower() not in ("0", "false", "no")


# ── Embedding mode ────────────────────────────────────────────────────────────
# Set via EMB_MODE env var or auto-detected from available files.
# "concat"     — concat(img_emb, text_emb) (2048-dim)  [best: image dominates]
# "img_only"   — img_emb only (1024-dim)  [ablation: pure image signal]
# "detr"       — SAM3 DETR cross-attention output (256-dim)  [ablation]
# "text"       — text_emb only (1024-dim, legacy ablation)
#
# Using image embedding as the primary signal is motivated by the observation
# that the optimal backbone layer depends more on image content than concept text:
# the same concept on a cluttered vs. clean background requires different layers.
def detect_emb_mode():
    if os.path.exists(os.path.join(DATA_DIR, "img_embs.npy")):
        return "concat"  # concat(img, text) — strongest signal
    if os.path.exists(os.path.join(DATA_DIR, "detr_embs.npy")):
        return "detr"
    return "text"

EMB_MODE = os.environ.get("EMB_MODE", detect_emb_mode())


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data():
    cgf1_matrix = np.load(os.path.join(DATA_DIR, "cgf1_matrix.npy")) # [N, L]
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    layer_list = meta["layer_list"]
    splits     = meta.get("splits", {})

    # ── Choose router input embedding ────────────────────────────────────────
    if EMB_MODE == "concat":
        # Primary signal: concat(img_emb, text_emb).
        # Image embedding captures spatial/appearance features that drive which
        # backbone layer is optimal.  Text adds concept-level context.
        img_embs    = np.load(os.path.join(DATA_DIR, "img_embs.npy"))   # [N, 1024]
        text_embs   = np.load(os.path.join(DATA_DIR, "text_embs.npy"))  # [N, 1024]
        router_embs = np.concatenate([img_embs, text_embs], axis=1)     # [N, 2048]
        mode_label  = "concat(img_emb, text_emb) (2048-dim) — image-dominant"
    elif EMB_MODE == "img_only":
        # Ablation: pure image signal, no text.
        router_embs = np.load(os.path.join(DATA_DIR, "img_embs.npy"))   # [N, 1024]
        mode_label  = "img_emb only (1024-dim) — ablation"
    elif EMB_MODE == "detr":
        router_embs = np.load(os.path.join(DATA_DIR, "detr_embs.npy"))  # [N, 256]
        mode_label  = "DETR cross-attention (256-dim) — ablation"
    else:
        router_embs = np.load(os.path.join(DATA_DIR, "text_embs.npy"))  # [N, 1024]
        mode_label  = "text_emb only (1024-dim, legacy ablation)"

    N = router_embs.shape[0]
    train_idx = splits.get("train", list(range(int(N * 0.70))))
    val_idx   = splits.get("val",   list(range(int(N * 0.70), int(N * 0.90))))

    print(f"Loaded: {N} total samples, {len(layer_list)} layers")
    print(f"Layer list: {layer_list}")
    print(f"Split (70/20/10): {len(train_idx)} train  {len(val_idx)} val  "
          f"{N - len(train_idx) - len(val_idx)} test (reserved for eval)")
    print(f"Router input mode: {mode_label}")

    # ── Filter to "failed cases" ──────────────────────────────────────────────
    # The router only needs to work on samples where L32 already fails.
    # Training on easy cases (where L32 is already optimal) produces mode
    # collapse: the router learns to always predict L31/L32 which are the most
    # frequent oracle layers, and ignores the cases that actually need routing.
    #
    # We keep a sample only when:
    #   max_cgF1(all layers) - cgF1(L32) >= DELTA_THRESHOLD
    # meaning the oracle best layer is at least DELTA_THRESHOLD (default 5%)
    # better than L32 in mask quality.  These are the cases routing can fix.
    if FOCUS_FAILED:
        # Find which column in cgf1_matrix corresponds to layer 32.
        # layer_list is [17, 18, ..., 32]; L32 is the last entry by construction.
        l32_col = layer_list.index(32) if 32 in layer_list else (len(layer_list) - 1)
        max_cgf1  = cgf1_matrix.max(axis=1)      # best achievable per sample [N]
        l32_cgf1  = cgf1_matrix[:, l32_col]      # L32 quality per sample [N]
        delta     = max_cgf1 - l32_cgf1           # cgF1 gain from optimal routing
        focus_set = set(np.where(delta >= DELTA_THRESHOLD)[0])

        n_focused = len(focus_set)
        print(f"\nFailed-cases filter ON  (DELTA_THRESHOLD={DELTA_THRESHOLD:.2f}):")
        print(f"  {n_focused} / {N} samples have oracle > L32 by ≥{DELTA_THRESHOLD:.2f} cgF1")
        print(f"  Routing only matters for these {100*n_focused/N:.1f}% of samples.")

        train_idx = [i for i in train_idx if i in focus_set]
        val_idx   = [i for i in val_idx   if i in focus_set]
        print(f"  Filtered split: {len(train_idx)} train | {len(val_idx)} val\n")
    else:
        print(f"\nFailed-cases filter OFF  (FOCUS_FAILED=0 — using full dataset)\n")

    X = torch.tensor(router_embs, dtype=torch.float32)
    C = torch.tensor(cgf1_matrix, dtype=torch.float32)

    # Soft targets: normalize cgF1 per sample with temperature
    C_temp = C / TEMPERATURE
    soft = F.softmax(C_temp, dim=-1)   # [N, L]

    # Hard targets: argmax of cgF1 per sample
    hard = C.argmax(dim=-1)            # [N]  integer class indices

    X_train, soft_train, hard_train = X[train_idx], soft[train_idx], hard[train_idx]
    X_val,   soft_val,   hard_val   = X[val_idx],   soft[val_idx],   hard[val_idx]

    # Compute class weights from training set distribution (inverse frequency).
    # This prevents the router from collapsing to always predicting L31 (the most
    # frequent oracle layer even in the failed-cases subset).
    num_classes = len(layer_list)
    class_counts = torch.zeros(num_classes)
    for c in hard_train:
        class_counts[c.item()] += 1
    class_counts = class_counts.clamp(min=1)
    class_weights = (1.0 / class_counts)
    class_weights = class_weights / class_weights.sum() * num_classes  # normalize
    print(f"Class weights (inverse freq, normalized):")
    for i, (l, w) in enumerate(zip(layer_list, class_weights.tolist())):
        print(f"  L{l}: count={int(class_counts[i])}  weight={w:.3f}")

    return X_train, soft_train, hard_train, X_val, soft_val, hard_val, layer_list, class_weights


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    torch.manual_seed(SEED)

    X_train, soft_train, hard_train, X_val, soft_val, hard_val, layer_list, class_weights = load_data()
    n_train, n_val = len(X_train), len(X_val)

    train_ds = TensorDataset(X_train, soft_train, hard_train)
    val_ds   = TensorDataset(X_val,   soft_val,   hard_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    num_layers = len(layer_list)
    input_dim  = X_train.shape[1]   # 2048 for concat(text, img)
    router = CAPRRouter(input_dim=input_dim, num_layers=num_layers,
                        layer_list=layer_list).to(DEVICE)

    optimizer = torch.optim.AdamW(router.parameters(), lr=LR, weight_decay=1e-3)
    # Warmup 10 epochs → cosine decay to 0.  Warmup prevents early collapse
    # toward majority classes before class weights take effect.
    def lr_lambda(epoch):
        warmup = 10
        if epoch < warmup:
            return float(epoch + 1) / warmup
        t = (epoch - warmup) / max(1, EPOCHS - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ce_weights = class_weights.to(DEVICE) if USE_CLASS_WEIGHTS else None

    mode_names = {256: "DETR cross-attn", 2048: "img+text concat", 1024: "img-only or text-only"}
    print(f"\nTraining (70/20/10 split): {n_train} train | {n_val} val | test reserved")
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  LR: {LR}  Batch: {BATCH_SIZE}")
    print(f"Router input_dim: {input_dim}  ({mode_names.get(input_dim, 'custom')})")
    print(f"Loss: α={ALPHA} × KL + {1-ALPHA:.1f} × CE  (T={TEMPERATURE})")
    print(f"Class-weighted CE: {USE_CLASS_WEIGHTS}  (prevents collapse to dominant layers)\n")

    best_val_loss = float("inf")
    best_weights  = None
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────────
        router.train()
        t_loss = 0.0
        for xb, sb, hb in train_loader:
            xb, sb, hb = xb.to(DEVICE), sb.to(DEVICE), hb.to(DEVICE)
            logits = router.mlp(xb)             # [B, L] (before softmax)
            log_p  = F.log_softmax(logits, dim=-1)

            kl_loss = F.kl_div(log_p, sb, reduction="batchmean")
            ce_loss = F.cross_entropy(logits, hb, weight=ce_weights)
            loss    = ALPHA * kl_loss + (1 - ALPHA) * ce_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(xb)

        # ── Val ──────────────────────────────────────────────────────────────
        router.eval()
        v_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, sb, hb in val_loader:
                xb, sb, hb = xb.to(DEVICE), sb.to(DEVICE), hb.to(DEVICE)
                logits  = router.mlp(xb)
                log_p   = F.log_softmax(logits, dim=-1)
                kl_loss = F.kl_div(log_p, sb, reduction="batchmean")
                ce_loss = F.cross_entropy(logits, hb)
                v_loss += (ALPHA * kl_loss + (1 - ALPHA) * ce_loss).item() * len(xb)
                correct += (logits.argmax(dim=-1) == hb).sum().item()

        t_loss /= n_train
        v_loss /= n_val
        val_acc = 100 * correct / n_val
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        scheduler.step()

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_weights  = {k: v.clone() for k, v in router.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{EPOCHS}  "
                  f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  "
                  f"val_acc={val_acc:.1f}%  (best_val={best_val_loss:.4f})")

    # ── Save best checkpoint — include layer_list so load_router() infers it ─
    # We embed layer_list in the checkpoint under a non-MLP key so that
    # load_router() can reconstruct the correct CAPRRouter without guessing.
    save_dict = dict(best_weights)
    save_dict["_layer_list"] = torch.tensor(layer_list, dtype=torch.long)
    weights_path = os.path.join(OUT_DIR, "capr_router_weights.pt")
    torch.save(save_dict, weights_path)
    print(f"\nBest checkpoint saved: {weights_path}")
    print(f"  best_val_loss = {best_val_loss:.4f}")

    # ── Loss curve ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train loss")
    ax.plot(val_losses,   label="val loss", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("CAPR Router Training — KL + CE Loss"); ax.legend(); ax.grid(alpha=0.3)
    curve_path = os.path.join(OUT_DIR, "router_training_curve.png")
    fig.savefig(curve_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Training curve saved: {curve_path}")

    # ── Layer prediction distribution on val set ──────────────────────────────
    router.load_state_dict(best_weights)
    router.eval()
    pred_layers, true_layers = [], []
    with torch.no_grad():
        for xb, sb, hb in val_loader:
            xb = xb.to(DEVICE)
            pred_layers.extend(router.mlp(xb).argmax(dim=-1).cpu().tolist())
            true_layers.extend(hb.tolist())
    pred_layers = np.array(pred_layers)
    true_layers = np.array(true_layers)
    top1_acc = (pred_layers == true_layers).mean()
    within1  = (np.abs(pred_layers - true_layers) <= 1).mean()
    print(f"\nVal set: top-1 accuracy = {100*top1_acc:.1f}%  within-1 accuracy = {100*within1:.1f}%")
    print(f"Predicted layer distribution:")
    for i, l in enumerate(layer_list):
        cnt = (pred_layers == i).sum()
        print(f"  L{l} (blk {l-1:2d}): {cnt:4d} / {len(pred_layers)}")


if __name__ == "__main__":
    train()
