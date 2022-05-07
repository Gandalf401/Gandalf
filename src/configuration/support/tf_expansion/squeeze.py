"""
    squeeze层
"""

import tensorflow as tf


class Squeeze(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def build(self, input_shape):
        super(Squeeze, self).build(input_shape)

    def call(self, x):
        return tf.squeeze(x, self.dim)
