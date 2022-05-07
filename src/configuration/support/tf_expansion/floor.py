"""
    floor层
"""
import tensorflow as tf


class Floor(tf.keras.layers.Layer):
    def __init__(self):
        super(Floor, self).__init__()

    def build(self, input_shape):
        super(Floor, self).build(input_shape)

    def call(self, x):
        return tf.math.floor(x)
