"""
    ceil层
"""
import torch


class Ceil(torch.nn.Module):
    def __init__(self):
        super(Ceil, self).__init__()

    def forward(self, x):
        return torch.ceil(x)
