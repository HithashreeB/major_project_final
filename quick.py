import torch
from swin_backbone import SwinBackbone

x = torch.randn(1, 3, 640, 640)
backbone = SwinBackbone()
features = backbone(x)

for i, f in enumerate(features):
    print(f"Feature {i}: {f.shape}")
