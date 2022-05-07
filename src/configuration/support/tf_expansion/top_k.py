"""
    top_k层
"""
import tensorflow as tf


class TopK(tf.keras.layers.Layer):
    def __init__(self, k, output='indices'):
        super(TopK, self).__init__()
        self.k = k
        self.output =output
        if output == 'indices':
            self.sentence = 'tf.math.top_k(x, self.k).indices'
        elif output == 'values':
            self.sentence = 'tf.math.top_k(x, self.k).values'
        else:
            self.sentence = 'tf.math.top_k(x, self.k)'
    
    def build(self, input_shape):
        super(TopK, self).build(input_shape)

    def call(self, x):
        return eval(self.sentence)
