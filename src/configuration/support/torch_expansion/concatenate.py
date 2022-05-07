"""
 torch concatenate层
"""
import torch


class Concatenate(torch.nn.Module):
    def __init__(self, axis=-1):
        super(Concatenate, self).__init__()
        self.axis = axis

    def forward(self, x):
        if not isinstance(x, list):
            raise Exception('Input of layer Concatenate should be a list.')
        if len(x) < 2:
            raise Exception('Input of layer Concatenate should be no less than 2.')
        res = torch.cat((x[0], x[1]), self.axis)
        for i in range(2, len(x)):
            res = torch.cat((res, x[i]), self.axis)
        return res
