"""
    square层
"""
import tensorflow as tf


class Square(tf.keras.layers.Layer):
    def __init__(self):
        super(Square, self).__init__()

    def build(self, input_shape):
        super(Square, self).build(input_shape)

    def call(self, x):
        return tf.square(x)
