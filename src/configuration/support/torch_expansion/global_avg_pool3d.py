"""
    torch自定义globalavgpool3d层
"""
import torch


class GlobalAvgPool3d(torch.nn.Module):
    def __init__(self):
        super(GlobalAvgPool3d, self).__init__()
        self.adaptive_avg_pool3d = torch.nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.adaptive_avg_pool3d(x)
        return torch.flatten(x, 1)
