"""
    sqrt层
"""
import jittor


class Sqrt(jittor.nn.Module):
    def __init__(self):
        super(Sqrt, self).__init__()

    def execute(self, x):
        return jittor.sqrt(x)
