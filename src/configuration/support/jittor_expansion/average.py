"""
 jittor average层
"""
import jittor


class Average(jittor.nn.Module):
    def __init__(self):
        super(Average, self).__init__()

    def execute(self, x):
        num = len(x)
        if not isinstance(x, list):
            raise Exception('Input of layer Average should be a list.')
        if len(x) < 2:
            raise Exception('Input of layer Average should be no less than 2.')
        res = x[0] + x[1]
        for i in range(2, num):
            res = res + x[i]
        return res / num
