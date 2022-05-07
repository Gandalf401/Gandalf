"""
    exp层
"""
import torch


class Exp(torch.nn.Module):
    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        return torch.exp(x)
