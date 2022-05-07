import numpy as np
import tensorflow as tf
import torch
import jittor


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return input_shape
    if params.__contains__('alpha'):
        alpha = params['alpha']
    else:
        alpha = 1e-2
    if framework == 'TensorFlow':
        initializer = tf.keras.initializers.Constant(alpha)
        return tf.keras.layers.LeakyReLU(alpha), input_shape
    elif framework == 'PyTorch':
        return torch.nn.LeakyReLU(alpha), input_shape
    elif framework == 'Jittor':
        return jittor.nn.LeakyReLU(alpha), input_shape
    else:
        raise Exception('No support DL framework.')
