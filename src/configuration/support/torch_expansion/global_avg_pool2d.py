"""
    torch自定义globalavgpool2d层
"""
import torch


class GlobalAvgPool2d(torch.nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
        self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.adaptive_avg_pool2d(x)
        return torch.flatten(x, 1)
