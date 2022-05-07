"""
    depthwise_conv2d层
"""
import jittor


class DepthwiseConv2d(jittor.nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, dilation, depth_multiplier, bias):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise_conv2d = jittor.nn.Conv2d(in_channels, depth_multiplier * in_channels, kernel_size, stride,
                                                 padding, dilation, in_channels, bias)

    def execute(self, x):
        return self.depthwise_conv2d(x)
