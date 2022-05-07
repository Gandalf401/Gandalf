"""
    ceil层
"""
import jittor


class Ceil(jittor.nn.Module):
    def __init__(self):
        super(Ceil, self).__init__()

    def execute(self, x):
        return jittor.ceil(x)
