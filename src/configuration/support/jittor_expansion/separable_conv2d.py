"""
    separable_conv2d层
"""
import jittor


class SeparableConv2d(jittor.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, depth_multiplier, bias):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = jittor.nn.Conv2d(in_channels, depth_multiplier * in_channels, kernel_size, stride,
                                                 padding, dilation, in_channels, False)
        self.conv1x1 = jittor.nn.Conv2d(depth_multiplier * in_channels, out_channels, 1, 1, 0, 1, 1, bias)

    def execute(self, x):
        x = self.depthwise_conv2d(x)
        return self.conv1x1(x)