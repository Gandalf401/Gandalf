"""
    gaussian_noise层
"""
import torch


class GaussianNoise(torch.nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, x):
        noise = torch.Tensor(x.shape)
        noise = torch.nn.init.normal(noise, 0.0, self.stddev).cuda()
        return x + noise

