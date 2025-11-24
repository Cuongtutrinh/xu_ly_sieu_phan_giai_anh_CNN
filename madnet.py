import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class RMAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.PReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.PReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=3, dilation=3),
            nn.PReLU()
        )
        self.ca = ChannelAttention(channels * 3)
        self.conv1x1 = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x + b1)
        b3 = self.branch3(x + b1 + b2)
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.ca(out)
        return self.conv1x1(out) + x

class MADNet(nn.Module):
    def __init__(self, upscale=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.PReLU()
        )
        self.drpbs = nn.ModuleList([RMAM(64) for _ in range(3)])
        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 4, 64, 1),
            nn.PReLU()
        )
        self.up = nn.Sequential(
            nn.Conv2d(64, 64 * upscale**2, 3, padding=1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        features = [x]
        for drpb in self.drpbs:
            x = drpb(x)
            features.append(x)
        x = self.fusion(torch.cat(features, dim=1))
        return self.up(x)