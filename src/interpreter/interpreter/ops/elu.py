import tensorflow as tf
import torch
import jittor


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return input_shape
    if params.__contains__('alpha'):
        alpha = params['alpha']
    else:
        alpha = 1.0
    if framework == 'TensorFlow':
        return tf.keras.layers.ELU(alpha), input_shape
    elif framework == 'PyTorch':
        return torch.nn.ELU(alpha), input_shape
    elif framework == 'Jittor':
        return jittor.nn.ELU(alpha), input_shape
    else:
        raise Exception('No support DL framework.')
