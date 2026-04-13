"""
CAPR Router — Concept-Adaptive Presence Routing (MoE style).

The router is a lightweight MLP that maps a text embedding to a distribution
over backbone layers. Two routing modes are supported:

  SOFT (MoE):  weighted sum of all layer features → FPN neck
               feature = sum_k( w_k * layer_norm(hidden_states[k]) )
               This is a true Mixture-of-Experts blend.

  HARD (top-1): pick the single layer with the highest weight → FPN neck
               feature = layer_norm(hidden_states[argmax(w)])

Architecture:
    text_emb (1024) → Linear(256) → LayerNorm → ReLU → Dropout(0.1)
                    → Linear(128) → ReLU
                    → Linear(num_layers) → Softmax

Weights:
    Pre-trained 17-layer weights are in results/capr_router_weights.pt
    (trained on layers [0,2,4,...,30,32]).
    For 33-layer routing, retrain with train_router.py.
"""
import os
import torch
import torch.nn as nn
from typing import List, Tuple

# The 17 layers the pre-trained router was trained on
ROUTER_LAYERS_17 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

# All 33 backbone layers (for full MoE)
ROUTER_LAYERS_33 = list(range(33))

PRETRAINED_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "results", "capr_router_weights.pt"
)


class CAPRRouter(nn.Module):
    """
    MoE-style router: text embedding → softmax weights over backbone layers.

    Args:
        text_dim   : input text embedding dimension (1024 for SAM3)
        num_layers : number of layers to route over (17 or 33)
        layer_list : the actual layer indices (e.g. [0,2,4,...,32] or [0,1,...,32])
    """

    def __init__(
        self,
        text_dim: int = 1024,
        num_layers: int = len(ROUTER_LAYERS_17),
        layer_list: List[int] = None,
    ):
        super().__init__()
        self.layer_list = layer_list or ROUTER_LAYERS_17
        assert len(self.layer_list) == num_layers
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_layers),
        )

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_emb : (batch, text_dim)
        Returns:
            weights  : (batch, num_layers)  softmax probabilities
        """
        return torch.softmax(self.mlp(text_emb), dim=-1)

    def get_layer_weights(self, text_emb: torch.Tensor) -> List[Tuple[int, float]]:
        """
        Returns list of (layer_idx, weight) sorted by weight descending.
        Use for soft MoE (take all) or hard routing (take first).
        """
        text_emb = text_emb.to(next(self.parameters()).device)
        weights = self.forward(text_emb).squeeze(0)  # (num_layers,)
        pairs = sorted(
            zip(self.layer_list, weights.tolist()),
            key=lambda x: x[1], reverse=True
        )
        return pairs  # [(layer_idx, weight), ...]

    def hard_pick(self, text_emb: torch.Tensor) -> int:
        """Return the single best layer index (top-1 hard routing)."""
        pairs = self.get_layer_weights(text_emb)
        return pairs[0][0]

    def top_k_weights(self, text_emb: torch.Tensor, k: int) -> List[Tuple[int, float]]:
        """Return top-k (layer_idx, weight) pairs, weights renormalized to sum=1."""
        pairs = self.get_layer_weights(text_emb)[:k]
        total = sum(w for _, w in pairs)
        return [(l, w / total) for l, w in pairs]


def load_router(weights_path: str = None) -> CAPRRouter:
    """
    Load a trained CAPRRouter. Infers layer list from saved weight shapes.
    If no weights found, returns a fresh untrained router with 17 layers.
    """
    if weights_path is None:
        weights_path = PRETRAINED_WEIGHTS

    if not os.path.exists(weights_path):
        print(f"WARNING: No router weights at {weights_path}.")
        print("         Returning UNTRAINED router (17 layers). Train with train_router.py")
        return CAPRRouter(text_dim=1024, num_layers=17, layer_list=ROUTER_LAYERS_17)

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    text_dim   = ckpt["mlp.0.weight"].shape[1]
    num_layers = ckpt["mlp.6.weight"].shape[0]

    # Prefer explicit layer_list saved in checkpoint (new format)
    if "_layer_list" in ckpt:
        layer_list = ckpt["_layer_list"].tolist()
    elif num_layers == 17:
        layer_list = ROUTER_LAYERS_17
    elif num_layers == 33:
        layer_list = ROUTER_LAYERS_33
    else:
        layer_list = list(range(num_layers))  # last-resort fallback

    model_state = {k: v for k, v in ckpt.items() if not k.startswith("_")}
    router = CAPRRouter(text_dim=text_dim, num_layers=num_layers, layer_list=layer_list)
    router.load_state_dict(model_state)
    router.eval()
    print(f"Router loaded: text_dim={text_dim}, num_layers={num_layers}, "
          f"layers={layer_list[:4]}...{layer_list[-1]}")
    return router
