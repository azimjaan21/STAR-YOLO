from models.starnet_backbone import StarNetBackboneForYOLO
import torch

model = StarNetBackboneForYOLO()
x = torch.randn(1, 3, 640, 640)
feats = model(x)
for i, f in enumerate(feats):
    print(f"P{i+3} shape:", f.shape)
