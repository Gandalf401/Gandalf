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
    if params.__contains__('dropout'):
        dropout = params['dropout']
    else:
        dropout = 0.0
    if shape_only:
        return [input_shape[0], hidden_size]
    if framework == 'TensorFlow':
        return tf.keras.layers.GRU(units=hidden_size, use_bias=bias, dropout=dropout, return_sequences=True), \
               [input_shape[0], hidden_size]
    elif framework == 'PyTorch':
        return torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, dropout=dropout, bias=bias,
                            batch_first=True), [input_shape[0], hidden_size]
    elif framework == 'Jittor':
        return jittor.nn.GRU(input_size=input_size, hidden_size=hidden_size, dropout=dropout, bias=bias,
                             batch_first=True), [input_shape[0], hidden_size]
    else:
        raise Exception('No support DL framework.')