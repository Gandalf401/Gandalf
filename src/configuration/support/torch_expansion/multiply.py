"""
 torch multiply层
"""
import torch


class Multiply(torch.nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, x):
        if not isinstance(x, list):
            raise Exception('Input of layer Multiply should be a list.')
        if len(x) < 2:
            raise Exception('Input of layer Multiply should be no less than 2.')
        res = x[0] * x[1]
        for i in range(2, len(x)):
            res = res * x[i]
        return res