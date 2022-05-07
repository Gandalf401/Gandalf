import tensorflow as tf
import torch
import jittor


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    input_size = params['input_size']
    hidden_size = params['hidden_size']
    if params.__contains__('bias'):
        bias = params['bias']
    else:
        bias = True
    if shape_only:
        return [input_shape[0], hidden_size]
    if framework == 'TensorFlow':
        return tf.keras.layers.SimpleRNNCell(units=hidden_size, use_bias=bias), [input_shape[0], hidden_size]
    elif framework == 'PyTorch':
        return torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size, bias=bias), \
               [input_shape[0], hidden_size]
    elif framework == 'Jittor':
        return jittor.nn.RNNCell(input_size=input_size, hidden_size=hidden_size, bias=bias), \
               [input_shape[0], hidden_size]
    else:
        raise Exception('No support DL framework.')