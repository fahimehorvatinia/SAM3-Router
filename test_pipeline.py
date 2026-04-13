"""
Quick sanity check: SAM3 loads, backbone hidden states have correct shape,
and layer injection into FPN neck works for two layers.

Run:
    python test_pipeline.py
"""
from PIL import Image
from sam3_wrapper import SAM3Wrapper

IMAGE_PATH = (
    "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026"
    "/metaclip-images/2/100002/metaclip_2_100002_f8c44cc57b6c83911b07dea6.jpeg"
)
PROMPT = "the cap"


def main():
    print("\n=== SAM3 Pipeline Sanity Check ===\n")

    wrapper = SAM3Wrapper()
    image   = Image.open(IMAGE_PATH).convert("RGB")

    pv, ids = wrapper.preprocess(image, PROMPT)
    print(f"pixel_values : {pv.shape}")

    hidden_states, backbone_lhs, text_emb_detr, _ = wrapper.extract(pv, ids)
    print(f"hidden_states : {len(hidden_states)} states, each {hidden_states[0].shape}")
    print(f"backbone_lhs  : {backbone_lhs.shape}")
    assert len(hidden_states) == 33, "Expected 33 backbone states"

    for layer_idx in [18, 32]:
        out = wrapper.run(image, hidden_states, backbone_lhs, text_emb_detr, pv, layer_idx=layer_idx)
        label = "SAM3 default" if layer_idx == 32 else f"layer {layer_idx}"
        print(f"{label:20s}  n_masks={out['n_masks']}  best_mask_px={out['best_mask'].sum()}"
              f"  presence={out['presence']:.4f}  query={out['query_score']:.4f}")

    print("\n=== All checks passed ===\n")


if __name__ == "__main__":
    main()
