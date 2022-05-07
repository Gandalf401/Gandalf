"""
    reduce_mean层
"""

import torch


class ReduceMean(torch.nn.Module):
    def __init__(self, dim, keep_dims, tensor_space):
        super(ReduceMean, self).__init__()
        self.dim = dim
        self.keep_dims = keep_dims
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
        if self.dim is None:
            return torch.mean(x, self.keep_dims)
        if x.ndim == self.tensor_space:
            x = eval(self.encoder[self.tensor_space])
        x = torch.mean(x, self.dim, self.keep_dims)
        if x.ndim == self.tensor_space:
            x = eval(self.decoder[self.tensor_space])
        return x
