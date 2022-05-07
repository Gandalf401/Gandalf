"""
    exp层
"""
import tensorflow as tf


class Exp(tf.keras.layers.Layer):
    def __init__(self):
        super(Exp, self).__init__()
    
    def build(self, input_shape):
        super(Exp, self).build(input_shape)

    def call(self, x):
        return tf.exp(x)
