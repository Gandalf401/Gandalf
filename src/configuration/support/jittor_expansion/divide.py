"""
 jittor divide层
"""
import jittor


class Divide(jittor.nn.Module):
    def __init__(self):
        super(Divide, self).__init__()

    def execute(self, x):
        if not isinstance(x, list):
            raise Exception('Input of layer Divide should be a list.')
        if len(x) < 2:
            raise Exception('Input of layer Divide should be no less than 2.')
        return jittor.divide(x[0], x[1])
