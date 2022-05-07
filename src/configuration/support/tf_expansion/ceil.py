"""
    ceil层
"""
import tensorflow as tf


class Ceil(tf.keras.layers.Layer):
    def __init__(self):
        super(Ceil, self).__init__()

    def build(self, input_shape):
        super(Ceil, self).build(input_shape)

    def call(self, x):
        return tf.math.ceil(x)
