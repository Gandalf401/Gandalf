"""
    gaussian_noise层
"""
import jittor
from jittor.init import gauss


class GaussianNoise(jittor.nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def execute(self, x):
        noise = gauss(x.shape, x.dtype, 0.0, self.stddev)
        return x + noise
