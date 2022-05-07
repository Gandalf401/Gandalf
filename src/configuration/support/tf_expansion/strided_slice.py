"""
    slice层
"""
import tensorflow as tf


class StridedSlice(tf.keras.layers.Layer):
    def __init__(self, begin, end, stride):
        super(StridedSlice, self).__init__()
        self.begin = begin
        self.end = end
        self.stride = stride

    def build(self, input_shape):
        super(StridedSlice, self).build(input_shape)

    def call(self, x):
        return tf.strided_slice(x, self.begin, self.end, self.stride)
