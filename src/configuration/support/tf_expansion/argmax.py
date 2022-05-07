"""
    argmax层
"""

import tensorflow as tf


class Argmax(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(Argmax, self).__init__()
        self.dim = dim

    def build(self, input_shape):
        super(Argmax, self).build(input_shape)

    def call(self, x):
        return tf.argmax(x, self.dim)
