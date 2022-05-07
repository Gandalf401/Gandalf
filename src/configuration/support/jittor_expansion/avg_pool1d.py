"""
    jittor下的avg pool1d扩展
"""
import jittor


class AvgPool1d(jittor.nn.Module):
    def __init__(self, pool_size=2, stride=1, padding=0):
        super(AvgPool1d, self).__init__()
        self.avg_pool2d = jittor.nn.AvgPool2d((pool_size, 1), (stride, 1), (padding, 0))

    def execute(self, x):
        x = jittor.reshape(x, [-1, x.shape[1], 1, x.shape[-1]])
        x = self.avg_pool2d(x)
        return jittor.reshape(x, [-1, x.shape[1], x.shape[-1]])

