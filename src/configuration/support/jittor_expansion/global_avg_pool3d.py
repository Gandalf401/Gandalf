"""
    jittor自定义globalavgpool3d层
"""
import jittor


class GlobalAvgPool3d(jittor.nn.Module):
    def __init__(self):
        super(GlobalAvgPool3d, self).__init__()
        self.adaptive_avg_pool3d = jittor.nn.AdaptiveAvgPool3d(1)

    def execute(self, x):
        x = self.adaptive_avg_pool3d(x)
        return jittor.flatten(x, 1)
