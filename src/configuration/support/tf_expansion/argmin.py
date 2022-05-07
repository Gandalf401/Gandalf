"""
    argmin层
"""

import tensorflow as tf


class Argmin(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(Argmin, self).__init__()
        self.dim = dim

    def build(self, input_shape):
        super(Argmin, self).build(input_shape)

    def call(self, x):
        return tf.argmin(x, self.dim)
