"""
    floor层
"""
import jittor


class Floor(jittor.nn.Module):
    def __init__(self):
        super(Floor, self).__init__()

    def execute(self, x):
        return jittor.floor(x)
