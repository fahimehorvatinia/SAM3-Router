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

EPOCHS     = 150
LR         = 1e-3
BATCH_SIZE = 64
# Split indices come from meta.json (70/20/10 created by collect_oracle_layers.py).
# Training uses the 70% train split; val uses the 20% val split.
# The 10% test split is reserved for eval_router_full.py.
ALPHA      = 0.6        # weight for soft KL loss (vs hard CE)
TEMPERATURE= 2.0        # softmax temperature for soft labels
SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data():
    text_embs   = np.load(os.path.join(DATA_DIR, "text_embs.npy"))    # [N, 1024]
    cgf1_matrix = np.load(os.path.join(DATA_DIR, "cgf1_matrix.npy")) # [N, L]
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    layer_list = meta["layer_list"]
    splits     = meta.get("splits", {})

    N = text_embs.shape[0]
    train_idx = splits.get("train", list(range(int(N * 0.70))))
    val_idx   = splits.get("val",   list(range(int(N * 0.70), int(N * 0.90))))
    # test_idx reserved for eval_router_full.py

    print(f"Loaded: {N} total samples, {len(layer_list)} layers")
    print(f"Layer list: {layer_list}")
    print(f"Split (70/20/10): {len(train_idx)} train  {len(val_idx)} val  "
          f"{N - len(train_idx) - len(val_idx)} test (reserved for eval)")

    X = torch.tensor(text_embs,   dtype=torch.float32)
    C = torch.tensor(cgf1_matrix, dtype=torch.float32)

    # Soft targets: normalize cgF1 per sample with temperature
    C_temp = C / TEMPERATURE
    soft = F.softmax(C_temp, dim=-1)   # [N, L]

    # Hard targets: argmax of cgF1 per sample
    hard = C.argmax(dim=-1)            # [N]  integer class indices

    X_train, soft_train, hard_train = X[train_idx], soft[train_idx], hard[train_idx]
    X_val,   soft_val,   hard_val   = X[val_idx],   soft[val_idx],   hard[val_idx]

    return X_train, soft_train, hard_train, X_val, soft_val, hard_val, layer_list


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    torch.manual_seed(SEED)

    X_train, soft_train, hard_train, X_val, soft_val, hard_val, layer_list = load_data()
    n_train, n_val = len(X_train), len(X_val)

    train_ds = TensorDataset(X_train, soft_train, hard_train)
    val_ds   = TensorDataset(X_val,   soft_val,   hard_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    num_layers = len(layer_list)
    router = CAPRRouter(text_dim=1024, num_layers=num_layers,
                        layer_list=layer_list).to(DEVICE)

    optimizer = torch.optim.Adam(router.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nTraining (70/20/10 split): {n_train} train | {n_val} val | test reserved")
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  LR: {LR}  Batch: {BATCH_SIZE}")
    print(f"Loss: α={ALPHA} × KL + {1-ALPHA:.1f} × CE  (T={TEMPERATURE})\n")

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
            ce_loss = F.cross_entropy(logits, hb)
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
