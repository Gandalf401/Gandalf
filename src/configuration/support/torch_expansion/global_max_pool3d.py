"""
    torch自定义globalmaxgpool3d层
"""
import torch


class GlobalMaxPool3d(torch.nn.Module):
    def __init__(self):
        super(GlobalMaxPool3d, self).__init__()
        self.adaptive_max_pool3d = torch.nn.AdaptiveMaxPool3d(1)

    def forward(self, x):
        x = self.adaptive_max_pool3d(x)
        return torch.flatten(x, 1)
