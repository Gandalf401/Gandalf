"""
    torch下causal padding
"""
import torch


class CausalPadding1d(torch.nn.Module):
    def __init__(self, pad):
        super(CausalPadding1d, self).__init__()
        self.pad = [pad, 0]

    def forward(self, x):
        return torch.nn.pad(x, self.pad)
