"""
 jittor minimum层
"""
import jittor


class Minimum(jittor.nn.Module):
    def __init__(self):
        super(Minimum, self).__init__()

    def execute(self, x):
        if not isinstance(x, list):
            raise Exception('Input of layer Minimum should be a list.')
        if len(x) < 2:
            raise Exception('Input of layer Minimum should be no less than 2.')
        res = jittor.minimum(x[0], x[1])
        for i in range(2, len(x)):
            res = jittor.minimum(x[i], res)
        return res
