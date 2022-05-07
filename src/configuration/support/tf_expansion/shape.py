"""
    shape层
"""
import tensorflow as tf


class Shape(tf.keras.layers.Layer):
    def __init__(self):
        super(Shape, self).__init__()

    def build(self, input_shape):
        super(Shape, self).build(input_shape)

    def call(self, x):
        return tf.shape(x)
