"""
    square层
"""
import jittor


class Square(jittor.nn.Module):
    def __init__(self):
        super(Square, self).__init__()

    def execute(self, x):
        return x * x
