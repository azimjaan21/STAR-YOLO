import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv  # reuse ultralytics Conv block

# Directly from official repo
class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

class StarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=False)
        self.act = nn.ReLU6()

    def forward(self, x):
        x = self.dwconv(x)
        x = self.act(self.f1(x)) * self.f2(x)
        x = self.dwconv2(self.g(x))
        return x

# Adapted Backbone for YOLO
class StarNetBackboneForYOLO(nn.Module):
    def __init__(self, mlp_ratio=3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBN(3, 32, 3, stride=2, padding=1),  # [B, 32, 320, 320]
            nn.ReLU6()
        )

        self.stage1 = nn.Sequential(
            ConvBN(32, 64, 3, 2, 1),       # downsample → [B, 64, 160, 160]
            StarBlock(64, mlp_ratio),
            StarBlock(64, mlp_ratio)
        )

        self.stage2 = nn.Sequential(
            ConvBN(64, 128, 3, 2, 1),      # → [B, 128, 80, 80]
            StarBlock(128, mlp_ratio),
            StarBlock(128, mlp_ratio)
        )

        self.stage3 = nn.Sequential(
            ConvBN(128, 256, 3, 2, 1),     # → [B, 256, 40, 40]
            StarBlock(256, mlp_ratio),
            StarBlock(256, mlp_ratio)
        )

        self.stage4 = nn.Sequential(
            ConvBN(256, 512, 3, 2, 1),     # → [B, 512, 20, 20]
            StarBlock(512, mlp_ratio),
            StarBlock(512, mlp_ratio)
        )

    def forward(self, x):
        x = self.stem(x)
        x1 = self.stage1(x)  # 160x160
        x2 = self.stage2(x1) # 80x80 → P3
        x3 = self.stage3(x2) # 40x40 → P4
        x4 = self.stage4(x3) # 20x20 → P5
        return [x2, x3, x4]
