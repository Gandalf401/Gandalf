"""
    tile层
"""
import tensorflow as tf


class Tile(tf.keras.layers.Layer):
    def __init__(self, multiples):
        super(Tile, self).__init__()
        self.multiples = multiples
    
    def build(self, input_shape):
        super(Tile, self).build(input_shape)

    def call(self, x):
        return tf.tile(x, self.multiples)
