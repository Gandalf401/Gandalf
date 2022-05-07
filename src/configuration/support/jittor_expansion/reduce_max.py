"""
    reduce_max层
"""

import jittor


class ReduceMax(jittor.nn.Module):
    def __init__(self, dim, keep_dims, tensor_space):
        super(ReduceMax, self).__init__()
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

    def execute(self, x):
        if self.dim is None:
            return jittor.max(x, self.keep_dims)
        if x.ndim == self.tensor_space:
            x = eval(self.encoder[self.tensor_space])
        x = jittor.max(x, self.dim, self.keep_dims)
        if x.ndim == self.tensor_space:
            x = eval(self.decoder[self.tensor_space])
        return x
