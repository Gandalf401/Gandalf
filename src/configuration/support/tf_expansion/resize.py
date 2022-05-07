"""
    resize层
"""
import tensorflow as tf


class Resize(tf.keras.layers.Layer):
    def __init__(self, size, mode):
        super(Resize, self).__init__()
        mode_table = {
            'bilinear': tf.image.ResizeMethod.BILINEAR,
            'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            'bicubic': tf.image.ResizeMethod.BICUBIC
        }
        self.size = size
        self.mode = mode_table[mode]

    def build(self, input_shape):
        super(Resize, self).build(input_shape)

    def call(self, x):
        return tf.image.resize(x, self.size, self.mode)
