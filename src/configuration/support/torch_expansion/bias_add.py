"""
 torch bias_add层
"""
import torch


class BiasAdd(torch.nn.Module):
    def __init__(self, bias):
        super(BiasAdd, self).__init__()
        self.bias = bias

    def forward(self, x):
        x_dtype = x.dtype
        x_shape = x.shape
        bias = torch.full(x_shape, self.bias, dtype=x_dtype).cuda()
        return x + bias
