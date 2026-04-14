"""
SAM3 Wrapper — real model, real inference, layer injection into FPN neck.

Layer naming convention (matches SAM3 architecture):
    hidden_states[0]    = patch embedding (before any transformer block)
    hidden_states[1]    = after transformer block 0
    hidden_states[k]    = after transformer block k-1   (k = 1..32)
    hidden_states[32]   = after transformer block 31  ← SAM3 default

DIAGNOSTIC CONFIRMED (2025):
    - hidden_states[32] is NOT pre-normed by the backbone.
      Max diff hs[32] vs layer_norm(hs[32]) = 125.16 → must apply LayerNorm to ALL layers.
    - backbone_out.last_hidden_state is NOT the same tensor as hidden_states[32].
      It has a different shape and must be stored separately from extract() and passed
      into Sam3VisionEncoderOutput.last_hidden_state.

Usage:
    wrapper = SAM3Wrapper()
    pixel_values, input_ids = wrapper.preprocess(image, "the cap")
    hidden_states, backbone_lhs, text_emb_detr, text_emb_router = wrapper.extract(pixel_values, input_ids)
    result = wrapper.run(image, hidden_states, backbone_lhs, text_emb_detr, pixel_values, layer_idx=18)
    # result["masks"]       — list of all above-threshold binary masks
    # result["best_mask"]   — single best mask (highest query score)
    # result["n_masks"]     — number of masks found above threshold
    # result["presence"]    — sigmoid(presence_logit)
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Tuple

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from transformers import Sam3Model, Sam3Processor
from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

SAM3_MODEL_ID = "facebook/sam3"

# All 33 hidden-state indices: 0 (patch embedding) through 32 (after block 31).
# Index 32 = SAM3 default (after the final transformer block 31).
SWEEP_LAYERS = list(range(33))      # [0, 1, 2, ..., 32]
SAM3_DEFAULT = 32                   # hidden_states[32] = after block 31 = SAM3 default
PATCH_SIZE   = 14
MASK_SCORE_THRESHOLD = 0.5          # query score threshold to count a mask as "found"


def layer_label(idx: int) -> str:
    """Human-readable label for a hidden_state index."""
    if idx == 0:
        return "embed"
    if idx == SAM3_DEFAULT:
        return f"block {idx-1} (SAM3 default)"
    return f"block {idx - 1}"


class SAM3Wrapper:
    """Loads real SAM3 once and supports per-layer FPN injection."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, local_files_only=True)
        self.model = Sam3Model.from_pretrained(SAM3_MODEL_ID, local_files_only=True).to(self.device)
        self.model.eval()
        print(f"SAM3 ready on {self.device}")

    def preprocess(self, image: Image.Image, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (pixel_values, input_ids) on device."""
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        return inputs["pixel_values"].to(self.device), inputs["input_ids"].to(self.device)

    @torch.no_grad()
    def extract(
        self, pixel_values: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[tuple, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run backbone + text encoder ONCE per image/prompt pair.

        Returns:
            hidden_states    : tuple of 33 tensors, each (1, 72, 72, 1024) on device.
                               NONE of these are pre-LayerNormed — run() applies it.
            backbone_lhs     : backbone_out.last_hidden_state — the tensor SAM3 normally
                               passes as last_hidden_state in Sam3VisionEncoderOutput.
                               This is NOT the same as hidden_states[32].
            text_emb_detr    : (1, seq, D) pooler_output used by SAM3 DETR head
            text_emb_router  : (1024,) mean-pooled text embedding used by the CAPR router
            img_emb_router   : (1024,) mean-pooled final-layer image embedding for routing.
                               Computed as spatial mean of hidden_states[32] (before LayerNorm).
                               Together with text_emb_router, forms the (2048,) router input.
        """
        backbone_out    = self.model.vision_encoder.backbone(pixel_values, output_hidden_states=True)
        text_out        = self.model.get_text_features(input_ids=input_ids, return_dict=True)
        text_emb_detr   = text_out.pooler_output
        text_emb_router = text_out.last_hidden_state.mean(dim=1).squeeze(0)   # (1024,)
        # Mean-pool hidden_states[32] over spatial dims (H=72, W=72) → (1024,)
        img_emb_router  = backbone_out.hidden_states[32].mean(dim=(1, 2)).squeeze(0)  # (1024,)
        return (
            backbone_out.hidden_states,       # tuple of 33 spatial tensors
            backbone_out.last_hidden_state,   # separate — passed to Sam3VisionEncoderOutput
            text_emb_detr,
            text_emb_router,
            img_emb_router,
        )

    def _build_output(
        self,
        spatial_features: torch.Tensor,
        backbone_lhs: torch.Tensor,
        hidden_states: tuple,
        text_emb_detr: torch.Tensor,
        image: Image.Image,
    ) -> dict:
        """
        Shared final stage: FPN neck → DETR → masks.
        spatial_features : (B, C, H, W) already LayerNormed and reshaped.
        """
        fpn_feats, fpn_pos = self.model.vision_encoder.neck(spatial_features)

        vision_embeds = Sam3VisionEncoderOutput(
            last_hidden_state=backbone_lhs,
            fpn_hidden_states=fpn_feats,
            fpn_position_encoding=fpn_pos,
            hidden_states=hidden_states,
            attentions=None,
        )

        outputs = self.model(vision_embeds=vision_embeds, text_embeds=text_emb_detr)

        orig_h, orig_w = image.size[1], image.size[0]
        presence     = float(outputs.presence_logits.sigmoid().item())
        query_scores = outputs.pred_logits.sigmoid().squeeze(0).cpu()   # (200,)

        # Collect ALL masks whose query score exceeds threshold
        masks = []
        above = (query_scores > MASK_SCORE_THRESHOLD).nonzero(as_tuple=True)[0]
        for q in above.tolist():
            ml = outputs.pred_masks[0, q].cpu().float()
            mu = F.interpolate(
                ml.unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w), mode="bilinear", align_corners=False,
            ).squeeze()
            masks.append((mu.sigmoid() > 0.5).numpy().astype(bool))

        # Best mask = highest query score (always produced)
        best_q  = int(query_scores.argmax())
        best_ml = outputs.pred_masks[0, best_q].cpu().float()
        best_up = F.interpolate(
            best_ml.unsqueeze(0).unsqueeze(0),
            size=(orig_h, orig_w), mode="bilinear", align_corners=False,
        ).squeeze()
        best_mask = (best_up.sigmoid() > 0.5).numpy().astype(bool)

        # union_mask = union of only above-threshold masks.
        # best_mask is always returned separately for callers that need it
        # regardless of threshold, but is NOT forced into union_mask.
        union_mask = np.zeros_like(best_mask)
        for m in masks:
            union_mask |= m

        return {
            "masks":       masks,
            "best_mask":   best_mask,
            "union_mask":  union_mask,
            "n_masks":     len(masks),
            "presence":    presence,
            "query_score": float(query_scores[best_q]),
        }

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        hidden_states: tuple,
        backbone_lhs: torch.Tensor,
        text_emb_detr: torch.Tensor,
        pixel_values: torch.Tensor,
        layer_idx: int = SAM3_DEFAULT,
    ) -> dict:
        """
        Inject hidden_states[layer_idx] into the FPN neck, then run DETR + mask head.

        LayerNorm is applied to ALL layers (0-32) unconditionally — diagnostic confirmed
        that none of the hidden states come pre-normed from the backbone.

        Args:
            hidden_states : tuple of 33 tensors from extract()
            backbone_lhs  : backbone_out.last_hidden_state from extract()
            layer_idx     : hidden state index (0-32). 32 = SAM3 default final layer.
        """
        layer_idx = max(0, min(layer_idx, 32))
        target = hidden_states[layer_idx]   # (1, 72, 72, 1024)

        # Apply LayerNorm to ALL layers — none come pre-normed from the backbone.
        # (Confirmed: max diff hs[32] vs layer_norm(hs[32]) = 125.16)
        target = self.model.vision_encoder.backbone.layer_norm(target)

        B = target.shape[0]
        H = pixel_values.shape[-2] // PATCH_SIZE
        W = pixel_values.shape[-1] // PATCH_SIZE
        spatial = target.view(B, H, W, -1).permute(0, 3, 1, 2)   # (B, C, H, W)

        return self._build_output(spatial, backbone_lhs, hidden_states, text_emb_detr, image)

    @torch.no_grad()
    def extract_detr_emb(
        self,
        hidden_states: tuple,
        backbone_lhs: torch.Tensor,
        text_emb_detr: torch.Tensor,
        pixel_values: torch.Tensor,
        layer_idx: int = SAM3_DEFAULT,
    ) -> torch.Tensor:
        """
        Run FPN neck + DETR decoder for a given backbone layer and return the
        cross-attention fusion embedding for routing.

        This is the output of SAM3's own text-image cross-attention:
          detr_decoder.intermediate_hidden_states[-1]  →  (B, 200, 256)
          mean over 200 queries                        →  (256,)

        This is richer than concat(text_emb, img_emb) because SAM3 has already
        learned to align text concepts with image regions inside the DETR decoder.

        Args:
            layer_idx : which backbone layer to inject (default=32 for routing pass)
        Returns:
            detr_emb  : (256,) cross-modal fusion embedding
        """
        layer_idx = max(0, min(layer_idx, 32))
        target    = hidden_states[layer_idx]
        target    = self.model.vision_encoder.backbone.layer_norm(target)

        B  = target.shape[0]
        H  = pixel_values.shape[-2] // PATCH_SIZE
        W  = pixel_values.shape[-1] // PATCH_SIZE
        spatial = target.view(B, H, W, -1).permute(0, 3, 1, 2)

        fpn_feats, fpn_pos = self.model.vision_encoder.neck(spatial)

        # Register a one-time hook to capture intermediate_hidden_states
        captured = {}
        def _hook(module, inp, out):
            captured["dec"] = out
        handle = self.model.detr_decoder.register_forward_hook(_hook)

        # Pass hidden_states=None here — the DETR forward only needs fpn_hidden_states
        # and fpn_position_encoding; the raw hidden_states field is not consumed downstream.
        # Passing the real hs tuple would let SAM3 potentially modify those tensors
        # in-place, silently corrupting the caller's hs for all subsequent wrapper.run()
        # calls in the same sample loop.
        vision_embeds = Sam3VisionEncoderOutput(
            last_hidden_state=backbone_lhs,
            fpn_hidden_states=fpn_feats,
            fpn_position_encoding=fpn_pos,
            hidden_states=None,
            attentions=None,
        )
        self.model(vision_embeds=vision_embeds, text_embeds=text_emb_detr)
        handle.remove()

        # intermediate_hidden_states: (num_decoder_layers, B, num_queries, 256)
        # Use last decoder layer, mean-pool over 200 queries → (256,)
        dec_out = captured["dec"]
        detr_emb = dec_out.intermediate_hidden_states[-1].mean(dim=1).squeeze(0)  # (256,)
        return detr_emb.float()

    @torch.no_grad()
    def run_moe(
        self,
        image: Image.Image,
        hidden_states: tuple,
        backbone_lhs: torch.Tensor,
        text_emb_detr: torch.Tensor,
        pixel_values: torch.Tensor,
        layer_weights: list,
    ) -> dict:
        """
        MoE soft routing: weighted blend of multiple backbone layer features → FPN neck.

        feature = sum_k( w_k * layer_norm(hidden_states[layer_k]) )

        LayerNorm is applied to ALL layers unconditionally before blending.

        Args:
            layer_weights : list of (layer_idx, weight) — weights must sum to 1.
                            e.g. [(28, 0.6), (32, 0.4)]
        """
        B = hidden_states[0].shape[0]
        H = pixel_values.shape[-2] // PATCH_SIZE
        W = pixel_values.shape[-1] // PATCH_SIZE

        weighted = None
        for layer_idx, w in layer_weights:
            layer_idx = max(0, min(layer_idx, 32))
            # Apply LayerNorm to ALL layers unconditionally
            target = self.model.vision_encoder.backbone.layer_norm(hidden_states[layer_idx])
            weighted = target * w if weighted is None else weighted + target * w

        spatial = weighted.view(B, H, W, -1).permute(0, 3, 1, 2)   # (B, C, H, W)

        return self._build_output(spatial, backbone_lhs, hidden_states, text_emb_detr, image)
