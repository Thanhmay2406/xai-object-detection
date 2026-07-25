from torch import nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, bottom_up, min_level=2, max_level=6, out_channels=256):
        super().__init__()
        self.bottom_up = bottom_up
        in_channels = getattr(bottom_up, "out_channels", [256, 512, 1024, 2048])
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(channels, out_channels, kernel_size=1) for channels in in_channels
        )
        self.output_convs = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        )
        self.p6 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.min_level = min_level
        self.max_level = max_level

        for module in list(self.lateral_convs) + list(self.output_convs):
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = list(self.bottom_up(x).values())
        laterals = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]
        results = [None] * len(laterals)
        prev = laterals[-1]
        results[-1] = self.output_convs[-1](prev)
        for idx in range(len(laterals) - 2, -1, -1):
            top_down = F.interpolate(prev, size=laterals[idx].shape[-2:], mode="nearest")
            prev = laterals[idx] + top_down
            results[idx] = self.output_convs[idx](prev)

        p2, p3, p4, p5 = results
        p6 = self.p6(p5)
        return [p6, p5, p4, p3, p2], features
