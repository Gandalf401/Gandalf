"""
    cast层
"""
import tensorflow as tf


class Cast(tf.keras.layers.Layer):
    def __init__(self, t_dtype):
        super(Cast, self).__init__()
        dtype_table = {
            "float": tf.float32,
            "float32": tf.float32,
            "float64": tf.float64,
            "int": tf.int32,
            "int8": tf.int8,
            "int16": tf.int16,
            "int32": tf.int32,
            "int64": tf.int64
        }
        self.t_dtype = dtype_table[t_dtype]

    def build(self, input_shape):
        super(Cast, self).build(input_shape)

    def call(self, x):
        return tf.cast(x, self.t_dtype)
