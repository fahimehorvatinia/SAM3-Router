from sam3_wrapper import SAM3Wrapper
from PIL import Image
import numpy as np

wrapper = SAM3Wrapper()
img = Image.fromarray(np.random.randint(0, 255, (1008, 1008, 3), dtype=np.uint8))
pv, ids = wrapper.preprocess(img, "the cap")

backbone_out = wrapper.model.vision_encoder.backbone(pv, output_hidden_states=True)
hs = backbone_out.hidden_states

print(f"Number of hidden states: {len(hs)}")
print(f"hs[0].shape:  {hs[0].shape}")
print(f"hs[16].shape: {hs[16].shape}")
print(f"hs[32].shape: {hs[32].shape}")

hs_32 = hs[32]
hs_ln = wrapper.model.vision_encoder.backbone.layer_norm(hs_32)
diff = (hs_32 - hs_ln).abs().max().item()
print(f"\nMax diff hs[32] vs layer_norm(hs[32]): {diff:.6f}")
print("already normed" if diff < 0.01 else "NOT pre-normed — wrapper has a bug")

lhs = backbone_out.last_hidden_state
diff2 = (lhs - hs_32).abs().max().item()
print(f"Max diff last_hidden_state vs hs[32]: {diff2:.6f}")
print("last_hidden_state IS hs[32]" if diff2 < 0.001 else "last_hidden_state is NOT hs[32]")