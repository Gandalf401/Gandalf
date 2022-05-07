"""
    rsqrt层
"""
import numpy as np
import tensorflow as tf


class Rsqrt(tf.keras.layers.Layer):
    def __init__(self):
        super(Rsqrt, self).__init__()

    def build(self, input_shape):
        super(Rsqrt, self).build(input_shape)

    def call(self, x):
        return tf.math.rsqrt(x)

