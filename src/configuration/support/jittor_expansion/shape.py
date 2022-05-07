"""
    shape层
"""
import jittor


class Shape(jittor.nn.Module):
    def __init__(self):
        super(Shape, self).__init__()

    def execute(self, x):
        return x.shape
