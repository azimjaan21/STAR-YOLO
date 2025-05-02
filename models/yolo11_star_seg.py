import torch
import torch.nn as nn
from models.starnet_backbone import StarNetBackbone
from ultralytics.nn.modules.conv import Conv

class StarYOLOSegModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = StarNetBackbone()
        self.seg_head = nn.Sequential(
            Conv(512, 256, 3, 1),
            nn.Upsample(scale_factor=2),
            Conv(256, num_classes, 1, 1)
        )

    def forward(self, x):
        features = self.backbone(x)  # P5 for now
        out = self.seg_head(features[-1])  # shape: [B, C, H, W]
        return out
