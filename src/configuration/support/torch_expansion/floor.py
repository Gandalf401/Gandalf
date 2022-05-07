"""
    floor层
"""
import torch


class Floor(torch.nn.Module):
    def __init__(self):
        super(Floor, self).__init__()

    def forward(self, x):
        return torch.floor(x)
