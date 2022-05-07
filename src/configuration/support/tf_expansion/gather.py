"""
    gather层
"""
import tensorflow as tf


class Gather(tf.keras.layers.Layer):
    def __init__(self, dim, index):
        super(Gather, self).__init__()
        self.dim = dim
        self.index = index

    def build(self, input_shape):
        super(Gather, self).build(input_shape)

    def call(self, x):
        return tf.gather(x, self.index, axis=self.dim)
