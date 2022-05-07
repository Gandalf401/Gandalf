"""
    depthwise_conv2d层
"""
import tensorflow as tf


class DepthwiseConv2d(tf.keras.layers.Layer):
    def __init__(self, filters, strides, padding, dilation):
        super(DepthwiseConv2d, self).__init__()
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.dilation = dilation

    def build(self, input_shape):
        super(DepthwiseConv2d, self).build(input_shape)

    def call(self, x):
        return tf.nn.depthwise_conv2d(x, self.filters, self.strides, self.padding, dilations=self.dilation)
