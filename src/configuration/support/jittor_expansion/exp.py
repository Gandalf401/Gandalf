"""
    exp层
"""
import jittor


class Exp(jittor.nn.Module):
    def __init__(self):
        super(Exp, self).__init__()

    def execute(self, x):
        return jittor.exp(x)
