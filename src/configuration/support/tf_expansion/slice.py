"""
    slice层
"""
import tensorflow as tf


class Slice(tf.keras.layers.Layer):
    def __init__(self, begin, size):
        super(Slice, self).__init__()
        self.begin = begin
        self.size = size

    def build(self, input_shape):
        super(Slice, self).build(input_shape)

    def call(self, x):
        return tf.slice(x, self.begin, self.size)
