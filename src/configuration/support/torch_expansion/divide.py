"""
 torch divide层
"""
import torch


class Divide(torch.nn.Module):
    def __init__(self):
        super(Divide, self).__init__()

    def forward(self, x):
        if not isinstance(x, list):
            raise Exception('Input of layer Divide should be a list.')
        if len(x) < 2:
            raise Exception('Input of layer Divide should be no less than 2.')
        return torch.divide(x[0], x[1])
