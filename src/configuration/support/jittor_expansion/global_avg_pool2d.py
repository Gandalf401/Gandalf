"""
    jittor自定义globalavgpool2d层
"""
import jittor


class GlobalAvgPool2d(jittor.nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
        self.adaptive_avg_pool2d = jittor.nn.AdaptiveAvgPool2d(1)

    def execute(self, x):
        x = self.adaptive_avg_pool2d(x)
        return jittor.flatten(x, 1)
