"""
    argmax层
"""

import torch


class Argmax(torch.nn.Module):
    def __init__(self, dim, channel_stay=False):
        super(Argmax, self).__init__()
        self.dim = dim
        self.channel_stay = channel_stay

    def forward(self, x):
        if self.channel_stay:
            if self.dim == -1 or self.dim == x.ndim:
                return torch.argmax(x, 1)
            elif self.dim == 0:
                return torch.argmax(x, 0)
            else:
                return torch.argmax(x, self.dim + 1)
        else:
            return torch.argmax(x, self.dim)
