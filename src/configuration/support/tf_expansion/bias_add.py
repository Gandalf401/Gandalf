"""
    bias_add层
"""
import numpy as np
import tensorflow as tf


class BiasAdd(tf.keras.layers.Layer):
    def __init__(self, bias):
        super(BiasAdd, self).__init__()
        self.bias = bias

    def build(self, input_shape):
        super(BiasAdd, self).build(input_shape)

    def call(self, x):
        bias = tf.fill([x.shape[-1]], self.bias)
        bias = tf.cast(bias, x.dtype)
        return tf.nn.bias_add(x, bias)
