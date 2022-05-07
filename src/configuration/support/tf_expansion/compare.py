"""
    compare层 各类张量元素级比较的封装层
"""
import tensorflow as tf


class Compare(tf.keras.layers.Layer):
    def __init__(self, op):
        super(Compare, self).__init__()
        op_table = {
            ">": "tf.greater(x, y)",
            ">=": "tf.greater_equal(x, y)",
            "==": "tf.equal(x, y)",
            "<=": "tf.less_equal(x, y)",
            "<": "tf.less(x, y)",
        }
        self.op = op_table[op]

    def build(self, input_shape):
        super(Compare, self).build(input_shape)

    def call(self, x, y):
        return eval(self.op)