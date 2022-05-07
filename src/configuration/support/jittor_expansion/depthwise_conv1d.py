"""
    depthwise_conv1d层
"""
import jittor


class DepthwiseConv1d(jittor.nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, dilation, depth_multiplier, bias):
        super(DepthwiseConv1d, self).__init__()
        self.depthwise_conv1d = jittor.nn.Conv1d(in_channels, depth_multiplier * in_channels, kernel_size, stride,
                                                 padding, dilation, in_channels, bias)

    def execute(self, x):
        return self.depthwise_conv1d(x)
