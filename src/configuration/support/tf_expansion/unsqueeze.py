"""
    unsqueeze层
"""

import tensorflow as tf


class Unsqueeze(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def build(self, input_shape):
        super(Unsqueeze, self).build(input_shape)

    def call(self, x):
        return tf.expand_dims(x, self.dim)
