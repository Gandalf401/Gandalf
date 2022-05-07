"""
    jittor自定义globalavgpool1d层
"""
import jittor


class GlobalAvgPool1d(jittor.nn.Module):
    def __init__(self):
        super(GlobalAvgPool1d, self).__init__()
        self.adaptive_avg_pool2d = jittor.nn.AdaptiveAvgPool2d(1)

    def execute(self, x):
        x = jittor.unsqueeze(x, 1)
        x = self.adaptive_avg_pool2d(x)
        return jittor.flatten(x, 1)
