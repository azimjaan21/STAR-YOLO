import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from models.starnet_backbone import StarNetBackboneForYOLO


class StarYOLOSegModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = StarNetBackboneForYOLO()

        self.seg_head = nn.Sequential(
            Conv(512, 256, 3, 1),
            nn.Upsample(scale_factor=2),   # 20 → 40
            Conv(256, 128, 3, 1),
            nn.Upsample(scale_factor=2),   # 40 → 80
            Conv(128, 64, 3, 1),
            nn.Upsample(scale_factor=2),   # 80 → 160
            Conv(64, num_classes, 1, 1)    # logits
        )

    def forward(self, x):
        _, _, H, W = x.shape
        feats = self.backbone(x)
        out = self.seg_head(feats[-1])  # P5 = [B, 512, 20, 20]
        out = nn.functional.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out  # shape: [B, 2, 640, 640]
