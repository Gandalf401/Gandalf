"""
 pytorch subtract层
"""
import torch


class Subtract(torch.nn.Module):
    def __init__(self):
        super(Subtract, self).__init__()

    def forward(self, x):
        if not isinstance(x, list):
            raise Exception('Input of layer Subtract should be a list.')
        if len(x) < 2:
            raise Exception('Input of layer Subtract should be no less than 2.')
        return torch.subtract(x[0], x[1])
