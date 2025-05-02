import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv  

class StarBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2):
        super().__init__()
        mid = in_channels * expansion
        self.conv1 = Conv(in_channels, mid, 1)
        self.dwconv = nn.Conv2d(mid, mid, 3, padding=1, groups=mid)
        self.fc1 = nn.Linear(mid, mid)
        self.fc2 = nn.Linear(mid, mid)
        self.relu = nn.ReLU(inplace=True)
        self.out = Conv(mid, out_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv1(x)
        x = self.dwconv(x)
        x_flat = x.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
        star = self.fc1(x_flat) * self.fc2(x_flat)
        star = self.relu(star)
        star = star.permute(0, 2, 1).view(b, -1, h, w)
        return self.out(star)

class StarNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = StarBlock(3, 32)
        self.stage2 = StarBlock(32, 64)
        self.stage3 = StarBlock(64, 128)
        self.stage4 = StarBlock(128, 256)
        self.stage5 = StarBlock(256, 512)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return [x3, x4, x5]  # For neck / segmentation head
