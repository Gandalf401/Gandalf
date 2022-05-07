"""
    transpose层
"""

import tensorflow as tf


class Transpose(tf.keras.layers.Layer):
    def __init__(self, output_shape):
        super(Transpose, self).__init__()
        self.target_shape = output_shape

    def build(self, input_shape):
        super(Transpose, self).build(input_shape)

    def call(self, x):
        return tf.transpose(x, self.target_shape)
