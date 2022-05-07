"""
    selu层
"""
import tensorflow as tf


class SeLU(tf.keras.layers.Layer):
    def __init__(self):
        super(SeLU, self).__init__()

    def build(self, input_shape):
        super(SeLU, self).build(input_shape)

    def call(self, x):
        return tf.nn.selu(x)
