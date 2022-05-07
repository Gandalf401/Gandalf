"""
    reduce_sum层
"""

import tensorflow as tf


class ReduceSum(tf.keras.layers.Layer):
    def __init__(self, dim, keep_dims):
        super(ReduceSum, self).__init__()
        self.dim = dim
        self.keep_dims = keep_dims

    def build(self, input_shape):
        super(ReduceSum, self).build(input_shape)

    def call(self, x):
        return tf.reduce_sum(x, self.dim, self.keep_dims)
