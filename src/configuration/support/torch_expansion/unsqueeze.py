"""
    unsqueeze层
"""

import torch


class Unsqueeze(torch.nn.Module):
    def __init__(self, dim, tensor_space):
        super(Unsqueeze, self).__init__()
        self.dim = dim
        self.tensor_space = tensor_space
        self.encoder = {
            3: 'x.permute(0, 2, 1)',
            4: 'x.permute(0, 2, 3, 1)',
            5: 'x.permute(0, 2, 3, 4, 1)'
        }
        self.decoder = {
            3: 'x.permute(0, 2, 1)',
            4: 'x.permute(0, 3, 1, 2)',
            5: 'x.permute(0, 4, 1, 2, 3)'
        }

    def forward(self, x):
        if x.ndim == self.tensor_space:
            x = eval(self.encoder[self.tensor_space])
        x = torch.unsqueeze(x, self.dim)
        if x.ndim == self.tensor_space:
            x = eval(self.decoder[self.tensor_space])
        return x
