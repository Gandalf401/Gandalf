"""
    torch自定义globalmaxpool1d层
"""
import torch


class GlobalMaxPool1d(torch.nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
        self.adaptive_max_pool1d = torch.nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.adaptive_max_pool1d(x)
        return torch.flatten(x, 1)
