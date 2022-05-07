"""
    squeeze层
"""

import jittor


class Squeeze(jittor.nn.Module):
    def __init__(self, dim, tensor_space):
        super(Squeeze, self).__init__()
        super(Squeeze, self).__init__()
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

    def execute(self, x):
        if x.ndim == self.tensor_space:
            x = eval(self.encoder[self.tensor_space])
        x = jittor.squeeze(x, self.dim)
        if x.ndim == self.tensor_space:
            x = eval(self.decoder[self.tensor_space])
        return x

