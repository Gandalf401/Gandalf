"""
    sqrt层
"""
import tensorflow as tf


class Sqrt(tf.keras.layers.Layer):
    def __init__(self):
        super(Sqrt, self).__init__()

    def build(self, input_shape):
        super(Sqrt, self).build(input_shape)

    def call(self, x):
        return tf.sqrt(x)
