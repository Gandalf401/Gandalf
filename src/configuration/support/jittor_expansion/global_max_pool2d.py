"""
    jittor自定义globalmaxgpool2d层
"""
import jittor


class GlobalMaxPool2d(jittor.nn.Module):
    def __init__(self):
        super(GlobalMaxPool2d, self).__init__()
        self.adaptive_max_pool2d = jittor.nn.AdaptiveMaxPool2d(1)

    def execute(self, x):
        x = self.adaptive_max_pool2d(x)
        return jittor.flatten(x, 1)
