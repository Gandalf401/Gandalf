"""
    reduce_prod层
"""

import tensorflow as tf


class ReduceProd(tf.keras.layers.Layer):
    def __init__(self, dim, keep_dims):
        super(ReduceProd, self).__init__()
        self.dim = dim
        self.keep_dims = keep_dims

    def build(self, input_shape):
        super(ReduceProd, self).build(input_shape)

    def call(self, x):
        return tf.reduce_prod(x, self.dim, self.keep_dims)
