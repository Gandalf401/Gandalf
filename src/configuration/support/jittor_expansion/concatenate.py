"""
 jittor concatenate层
"""
import jittor


class Concatenate(jittor.nn.Module):
    def __init__(self, axis=-1):
        super(Concatenate, self).__init__()
        self.axis = axis

    def execute(self, x):
        if not isinstance(x, list):
            raise Exception('Input of layer Concatenate should be a list.')
        if len(x) < 2:
            raise Exception('Input of layer Concatenate should be no less than 2.')
        res = jittor.cat((x[0], x[1]), self.axis)
        for i in range(2, len(x)):
            res = jittor.cat((res, x[i]), self.axis)
        return res
