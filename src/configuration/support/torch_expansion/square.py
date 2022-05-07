"""
    square层
"""
import torch


class Square(torch.nn.Module):
    def __init__(self):
        super(Square, self).__init__()

    def forward(self, x):
        return torch.square(x)
