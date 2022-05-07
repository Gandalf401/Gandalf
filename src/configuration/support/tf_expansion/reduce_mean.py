"""
    reduce_mean层
"""

import tensorflow as tf


class ReduceMean(tf.keras.layers.Layer):
    def __init__(self, dim, keep_dims):
        super(ReduceMean, self).__init__()
        self.dim = dim
        self.keep_dims = keep_dims

    def build(self, input_shape):
        super(ReduceMean, self).build(input_shape)

    def call(self, x):
        return tf.reduce_mean(x, self.dim, self.keep_dims)
