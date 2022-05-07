"""
    rsqrt层
"""
import torch


class Rsqrt(torch.nn.Module):
    def __init__(self):
        super(Rsqrt, self).__init__()

    def forward(self, x):
        return torch.rsqrt(x)
