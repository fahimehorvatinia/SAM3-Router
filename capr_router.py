"""
CAPR Router — Concept-Adaptive Presence Routing (MoE style).

Router input (v3 default):
    SAM3's own DETR decoder cross-attention output — 256-dim.
    Computed as: FPN(L32) + DETR decoder → last layer mean over 200 queries.

    This is the richest possible routing signal because:
    - SAM3's DETR decoder does interleaved text↔image cross-attention (6 layers)
    - Each query attends to BOTH text features and image (FPN) features
    - The output already encodes "how does this concept align with this image"
    - SAM3 was pre-trained to make this representation discriminative

Routing modes:
    HARD (top-1): pick the single layer with highest weight → one FPN+DETR pass
    SOFT (MoE):   weighted blend of all layer features → one FPN+DETR pass

Inference pipeline (two-pass):
    Backbone (once) → FPN(L32) + DETR → 256-dim routing emb
                                               ↓
                                          Router MLP → best layer k
                                               ↓
                      FPN(Lk)  + DETR → final masks + presence

MLP architecture per input_dim:
    256  (DETR, v3):    256 → 128 → 64 → num_layers   [compact, matches signal dim]
    2048 (concat, v2):  2048 → 512 → 256 → num_layers  [wider for raw concat]
    1024 (text, v1):    1024 → 256 → 128 → num_layers  [legacy ablation]
"""
import os
import torch
import torch.nn as nn
from typing import List, Tuple

ROUTER_LAYERS_16 = list(range(17, 33))          # 16 layers: 17..32  (current default)
ROUTER_LAYERS_17 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
ROUTER_LAYERS_33 = list(range(33))

PRETRAINED_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "results", "capr_router_weights.pt"
)

# MLP hidden sizes per input_dim
_MLP_SIZES = {
    256:  (128, 64),    # DETR cross-attn (v3)
    1024: (256, 128),   # text-only (v1 legacy)
    2048: (512, 256),   # concat text+img (v2)
}


class CAPRRouter(nn.Module):
    """
    MoE-style router: embedding → softmax weights over backbone layers.

    Args:
        input_dim  : 256 (DETR v3), 2048 (concat v2), 1024 (text v1)
        num_layers : number of candidate layers to route over
        layer_list : actual backbone layer indices, e.g. [17,18,...,32]
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_layers: int = len(ROUTER_LAYERS_16),
        layer_list: List[int] = None,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.layer_list = layer_list or ROUTER_LAYERS_16
        assert len(self.layer_list) == num_layers, \
            f"layer_list length {len(self.layer_list)} != num_layers {num_layers}"

        h1, h2 = _MLP_SIZES.get(input_dim, (max(64, input_dim // 4),
                                             max(32, input_dim // 8)))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LayerNorm(h1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_layers),
        )

    def forward(self, router_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            router_emb : (batch, input_dim)
        Returns:
            weights    : (batch, num_layers) softmax probabilities
        """
        return torch.softmax(self.mlp(router_emb), dim=-1)

    def get_layer_weights(self, router_emb: torch.Tensor) -> List[Tuple[int, float]]:
        """Returns [(layer_idx, weight), ...] sorted by weight descending."""
        router_emb = router_emb.to(next(self.parameters()).device)
        weights = self.forward(router_emb).squeeze(0)
        return sorted(zip(self.layer_list, weights.tolist()),
                      key=lambda x: x[1], reverse=True)

    def hard_pick(self, router_emb: torch.Tensor) -> int:
        """Return single best layer index (top-1 hard routing)."""
        return self.get_layer_weights(router_emb)[0][0]

    def top_k_weights(self, router_emb: torch.Tensor, k: int) -> List[Tuple[int, float]]:
        """Top-k (layer_idx, weight) pairs, weights renormalized to sum=1."""
        pairs = self.get_layer_weights(router_emb)[:k]
        total = sum(w for _, w in pairs)
        return [(l, w / total) for l, w in pairs]


def load_router(weights_path: str = None) -> CAPRRouter:
    """
    Load a trained CAPRRouter. Auto-detects input_dim and layer_list.
    Falls back to untrained DETR router (256-dim, 16 layers) if no weights found.
    """
    if weights_path is None:
        weights_path = PRETRAINED_WEIGHTS

    if not os.path.exists(weights_path):
        print(f"WARNING: No router weights at {weights_path}.")
        print("         Returning UNTRAINED router (DETR 256-dim, 16 layers).")
        return CAPRRouter(input_dim=256, num_layers=16, layer_list=ROUTER_LAYERS_16)

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)

    input_dim  = ckpt["mlp.0.weight"].shape[1]
    num_layers = ckpt["mlp.6.weight"].shape[0]

    if "_layer_list" in ckpt:
        layer_list = ckpt["_layer_list"].tolist()
    elif num_layers == 16:
        layer_list = ROUTER_LAYERS_16
    elif num_layers == 17:
        layer_list = ROUTER_LAYERS_17
    elif num_layers == 33:
        layer_list = ROUTER_LAYERS_33
    else:
        layer_list = list(range(num_layers))

    mode = {256: "DETR cross-attn (v3)", 2048: "text+img concat (v2)",
            1024: "text-only (v1)"}.get(input_dim, f"custom {input_dim}-dim")

    model_state = {k: v for k, v in ckpt.items() if not k.startswith("_")}
    router = CAPRRouter(input_dim=input_dim, num_layers=num_layers, layer_list=layer_list)
    router.load_state_dict(model_state)
    router.eval()
    print(f"Router loaded: {mode}, num_layers={num_layers}, "
          f"layers={layer_list[:4]}...{layer_list[-1]}")
    return router
