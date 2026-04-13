"""
Layer sweep: runs SAM3 inference with each backbone layer injected into the FPN neck.

Given one image and one text prompt, returns a list of results, one per layer.
This is the core diagnostic: which backbone layer gives the best concept grounding?

Usage:
    from layer_sweep import run_sweep
    results = run_sweep(wrapper, image, "the cap")
    # results is a list of dicts: [{layer_idx, presence, query_score, mask}, ...]
"""
from typing import List
from PIL import Image
from sam3_wrapper import SAM3Wrapper, SWEEP_LAYERS


def run_sweep(wrapper: SAM3Wrapper, image: Image.Image, prompt: str) -> List[dict]:
    """
    Run SAM3 with each layer in SWEEP_LAYERS injected into the FPN neck.

    The backbone and text encoder run ONCE. Only the FPN + DETR + mask head
    run once per layer. Fast: all layers share the same backbone features.

    Returns:
        List of dicts, one per layer (all 33: 0 … 32):
            layer_idx   : int   (backbone hidden state index)
            presence    : float (sigmoid of presence logit)
            query_score : float (max sigmoid of pred_logits)
            mask        : np.ndarray (H, W) bool
    """
    pixel_values, input_ids = wrapper.preprocess(image, prompt)

    # Backbone + text encoder run ONCE
    hidden_states, backbone_lhs, text_emb_detr, text_emb_router = wrapper.extract(pixel_values, input_ids)

    results = []
    for layer_idx in SWEEP_LAYERS:
        out = wrapper.run(image, hidden_states, backbone_lhs, text_emb_detr, pixel_values, layer_idx=layer_idx)
        results.append({
            "layer_idx":   layer_idx,
            "presence":    out["presence"],
            "query_score": out["query_score"],
            "mask":        out["best_mask"],
        })
        label = "final (SAM3 default)" if layer_idx == 32 else f"layer {layer_idx}"
        print(f"  {label:25s}  presence={out['presence']:.4f}  query={out['query_score']:.4f}"
              f"  mask_pixels={out['best_mask'].sum()}")

    return results, text_emb_router
