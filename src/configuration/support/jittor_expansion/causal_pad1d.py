"""
    jittor下causal padding
"""
import jittor


class CausalPadding1d(jittor.nn.Module):
    def __init__(self, pad):
        super(CausalPadding1d, self).__init__()
        self.pad = [pad, 0]

    def execute(self, x):
        return jittor.nn.pad(x, self.pad)
