"""
    torch自定义globalavgpool1d层
"""
import torch


class GlobalAvgPool1d(torch.nn.Module):
    def __init__(self):
        super(GlobalAvgPool1d, self).__init__()
        self.adaptive_avg_pool1d = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.adaptive_avg_pool1d(x)
        return torch.flatten(x, 1)
