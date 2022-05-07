"""
    bottleneck模块，用于branch_to和index
"""
import math

import tensorflow as tf


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, shape_needed, shape_r):
        super(Bottleneck, self).__init__()
        self.add = tf.keras.layers.Add()
        if shape_needed[0] >= shape_r[0]:
            h_diff = shape_needed[0] - shape_r[0]
            w_diff = shape_needed[1] - shape_r[1]
            self.pad = tf.keras.layers.ZeroPadding2D(padding=((h_diff // 2, math.ceil(h_diff / 2)),
                                                              (w_diff // 2, math.ceil(w_diff / 2))))
            if shape_needed[-1] == shape_r[-1]:
                self.conv11 = None
            else:
                self.conv11 = tf.keras.layers.Conv2D(shape_needed[-1], 1, 1)
        else:
            s = shape_r[0] // shape_needed[0] + 1
            h_diff = shape_needed[0] * s - shape_r[0]
            w_diff = shape_needed[1] * s - shape_r[1]
            self.pad = tf.keras.layers.ZeroPadding2D(padding=((h_diff // 2, math.ceil(h_diff / 2)),
                                                              (w_diff // 2, math.ceil(w_diff / 2))))
            self.conv11 = tf.keras.layers.Conv2D(shape_needed[-1], 1, s)

    def build(self, input_shape):
        super(Bottleneck, self).build(input_shape)

    def call(self, x):
        x, residual = x
        residual = self.pad(residual)
        if self.conv11 is not None:
            residual = self.conv11(residual)
        x = self.add([x, residual])
        return x
