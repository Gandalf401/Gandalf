"""
    jittor自定义globalmaxgpool3d层
"""
import jittor


class GlobalMaxPool3d(jittor.nn.Module):
    def __init__(self):
        super(GlobalMaxPool3d, self).__init__()
        self.adaptive_max_pool3d = jittor.nn.AdaptiveMaxPool3d(1)

    def execute(self, x):
        x = self.adaptive_max_pool3d(x)
        return jittor.flatten(x, 1)
