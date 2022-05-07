"""
    threshold层
"""
import jittor


class Threshold(jittor.nn.Module):
    def __init__(self, threshold):
        super(Threshold, self).__init__()
        self.threshold = threshold

    def execute(self, x):
        mask = (x < self.threshold).int()
        return jittor.masked_fill(x, mask, 0)
