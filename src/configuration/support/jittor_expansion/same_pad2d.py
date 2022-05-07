"""
    jittor下same padding的奇数pad情况扩展padding2d
"""
import jittor


class SamePadding2d(jittor.nn.Module):
    def __init__(self, height_pad, width_pad):
        super(SamePadding2d, self).__init__()
        self.pad = [0, int(width_pad % 2 != 0), 0, int(height_pad % 2 != 0)]

    def execute(self, x):
        return jittor.nn.pad(x, self.pad)
