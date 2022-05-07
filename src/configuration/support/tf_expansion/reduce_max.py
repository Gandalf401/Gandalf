"""
    reduce_max层
"""

import tensorflow as tf


class ReduceMax(tf.keras.layers.Layer):
    def __init__(self, dim, keep_dims):
        super(ReduceMax, self).__init__()
        self.dim = dim
        self.keep_dims = keep_dims

    def build(self, input_shape):
        super(ReduceMax, self).build(input_shape)

    def call(self, x):
        return tf.reduce_max(x, self.dim, self.keep_dims)
