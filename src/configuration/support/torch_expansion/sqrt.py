"""
    sqrt层
"""
import torch


class Sqrt(torch.nn.Module):
    def __init__(self):
        super(Sqrt, self).__init__()

    def forward(self, x):
        return torch.sqrt(x)
