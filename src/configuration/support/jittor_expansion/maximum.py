"""
 jittor maximum层
"""
import jittor


class Maximum(jittor.nn.Module):
    def __init__(self):
        super(Maximum, self).__init__()

    def execute(self, x):
        if not isinstance(x, list):
            raise Exception('Input of layer Maximum should be a list.')
        if len(x) < 2:
            raise Exception('Input of layer Maximum should be no less than 2.')
        res = jittor.maximum(x[0], x[1])
        for i in range(2, len(x)):
            res = jittor.maximum(x[i], res)
        return res
