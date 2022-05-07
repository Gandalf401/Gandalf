"""
    rsqrt层
"""
import jittor


class Rsqrt(jittor.nn.Module):
    def __init__(self):
        super(Rsqrt, self).__init__()

    def execute(self, x):
        ones = jittor.full(x.shape, 1, dtype=x.dtype)
        sqrt = jittor.sqrt(x)
        return jittor.divide(ones, sqrt)
