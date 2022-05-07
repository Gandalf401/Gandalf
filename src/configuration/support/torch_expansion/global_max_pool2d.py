"""
    torch自定义globalmaxgpool2d层
"""
import torch


class GlobalMaxPool2d(torch.nn.Module):
    def __init__(self):
        super(GlobalMaxPool2d, self).__init__()
        self.adaptive_max_pool2d = torch.nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.adaptive_max_pool2d(x)
        return torch.flatten(x, 1)
