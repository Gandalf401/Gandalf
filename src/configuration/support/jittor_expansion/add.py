"""
 jittor add层
"""
import jittor


class Add(jittor.nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def execute(self, x):
        if not isinstance(x, list):
            raise Exception('Input of layer Add should be a list.')
        if len(x) < 2:
            raise Exception('Input of layer Add should be no less than 2.')
        res = x[0] + x[1]
        for i in range(2, len(x)):
            res = res + x[i]
        return res
