"""
 tf divide层
"""
import tensorflow as tf


class Divide(tf.keras.layers.Layer):
    def __init__(self):
        super(Divide, self).__init__()
    
    def build(self, input_shape):
        super(Divide, self).build(input_shape)

    def call(self, x):
        if not isinstance(x, list):
            raise Exception('Input of layer Divide should be a list.')
        if len(x) < 2:
            raise Exception('Input of layer Divide should be no less than 2.')
        return tf.divide(x[0], x[1])
