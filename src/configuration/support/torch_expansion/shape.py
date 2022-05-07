"""
    shape层
"""
import torch


class Shape(torch.nn.Module):
    def __init__(self):
        super(Shape, self).__init__()

    def forward(self, x):
        return x.shape
