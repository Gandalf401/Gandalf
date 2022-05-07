"""
    separable_conv1d层
"""
import torch


class SeparableConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, depth_multiplier, bias):
        super(SeparableConv1d, self).__init__()
        self.depthwise_conv = torch.nn.Conv1d(in_channels, depth_multiplier * in_channels, kernel_size, stride,
                                                 padding, dilation, in_channels, False)
        self.conv1x1 = torch.nn.Conv1d(depth_multiplier * in_channels, out_channels, 1, 1, 0, 1, 1, bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        return self.conv1x1(x)
