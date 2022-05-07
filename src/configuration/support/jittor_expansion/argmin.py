"""
    argmin层
"""

import jittor


class Argmin(jittor.nn.Module):
    def __init__(self, dim, channel_stay=False):
        super(Argmin, self).__init__()
        self.dim = dim
        self.channel_stay = channel_stay

    def execute(self, x):
        if self.channel_stay:
            if self.dim == -1 or self.dim == x.ndim:
                return jittor.argmin(x, 1)
            elif self.dim == 0:
                return jittor.argmin(x, 0)
            else:
                return jittor.argmin(x, self.dim + 1)
        else:
            return jittor.argmin(x, self.dim)
