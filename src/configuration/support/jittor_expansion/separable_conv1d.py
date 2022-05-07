"""
    separable_conv1d层
"""
import jittor


class SeparableConv1d(jittor.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, depth_multiplier, bias):
        super(SeparableConv1d, self).__init__()
        self.depthwise_conv1d = jittor.nn.Conv1d(in_channels, depth_multiplier * in_channels, kernel_size, stride,
                                                 padding, dilation, in_channels, False)
        self.conv1x1 = jittor.nn.Conv1d(depth_multiplier * in_channels, out_channels, 1, 1, 0, 1, 1, bias)

    def execute(self, x):
        x = self.depthwise_conv1d(x)
        return self.conv1x1(x)