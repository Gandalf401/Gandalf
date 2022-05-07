"""
 jittor subtract层
"""
import jittor


class Subtract(jittor.nn.Module):
    def __init__(self):
        super(Subtract, self).__init__()

    def execute(self, x):
        if not isinstance(x, list):
            raise Exception('Input of layer Subtract should be a list.')
        if len(x) < 2:
            raise Exception('Input of layer Subtract should be no less than 2.')
        return jittor.subtract(x[0], x[1])
