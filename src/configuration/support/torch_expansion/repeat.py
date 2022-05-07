"""
    repeat层
"""
import torch


class Repeat(torch.nn.Module):
    def __init__(self, n):
        super(Repeat, self).__init__()
        self.n = n

    def forward(self, x):
        batch_size = x.size(0)
        x = x.repeat(1, self.n)
        x = torch.unsqueeze(x, 1)
        x = torch.reshape(x, (batch_size, self.n, -1))
        return x
