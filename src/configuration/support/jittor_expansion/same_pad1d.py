"""
    jittor下same padding的奇数pad情况扩展padding1d
"""
import jittor


class SamePadding1d(jittor.nn.Module):
    def __init__(self, pad):
        super(SamePadding1d, self).__init__()
        self.pad = [0, int(pad % 2 != 0)]

    def execute(self, x):
        return jittor.nn.pad(x, self.pad)
