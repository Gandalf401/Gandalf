"""
 torch maximum层
"""
import torch


class Maximum(torch.nn.Module):
    def __init__(self):
        super(Maximum, self).__init__()

    def forward(self, x):
        if not isinstance(x, list):
            raise Exception('Input of layer Maximum should be a list.')
        if len(x) < 2:
            raise Exception('Input of layer Maximum should be no less than 2.')
        res = torch.maximum(x[0], x[1])
        for i in range(2, len(x)):
            res = torch.maximum(x[i], res)
        return res
