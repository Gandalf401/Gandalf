"""
    jittor自定义globalmaxgpool1d层
"""
import jittor


class GlobalMaxPool1d(jittor.nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
        self.adaptive_max_pool2d = jittor.nn.AdaptiveMaxPool2d(1)

    def execute(self, x):
        x = jittor.unsqueeze(x, 1)
        x = self.adaptive_max_pool2d(x)
        return jittor.flatten(x, 1)
