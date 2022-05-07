"""
    argmax层
"""

import jittor


class Argmax(jittor.nn.Module):
    def __init__(self, dim, channel_stay=False):
        super(Argmax, self).__init__()
        self.dim = dim
        self.channel_stay = channel_stay

    def execute(self, x):
        if self.channel_stay:
            if self.dim == -1 or self.dim == x.ndim:
                return jittor.argmax(x, 1)
            elif self.dim == 0:
                return jittor.argmax(x, 0)
            else:
                return jittor.argmax(x, self.dim + 1)
        else:
            return jittor.argmax(x, self.dim)
