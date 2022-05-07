"""
 jittor bias_add层
"""
import jittor


class BiasAdd(jittor.nn.Module):
    def __init__(self, bias):
        super(BiasAdd, self).__init__()
        self.bias = bias

    def execute(self, x):
        x_dtype = x.dtype
        x_shape = x.shape
        bias = jittor.full(x_shape, self.bias, dtype=x_dtype)
        return x + bias



