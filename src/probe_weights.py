import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from paths import PATHS

for name in [PATHS.WEIGHTS_PREV, PATHS.WEIGHTS,
             os.path.join(PATHS.MODELS_DIR, "landslide_unet_weights.refine3ep.pth")]:
    try:
        s = torch.load(name, map_location="cpu", weights_only=True)
        w = s.get("down1.conv.0.weight")
        n = sum(v.numel() for v in s.values()) / 1e6
        bf = w.shape[0] if w is not None else "?"
        print(f"{name}: base_filters={bf}, {n:.1f}M params, keys={len(s)}")
    except Exception as e:
        print(f"{name}: ERROR {e}")
