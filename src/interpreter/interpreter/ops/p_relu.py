import numpy as np
import tensorflow as tf
import torch
import jittor


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return input_shape
    if params.__contains__('init'):
        alpha = params['init']
    else:
        alpha = 0.0
    if params.__contains__('share'):
        share = params['share']
    else:
        share = True
    if framework == 'TensorFlow':
        initializer = tf.keras.initializers.Constant(alpha)
        if share:
            shared = [i + 1 for i in range(len(input_shape))]
        else:
            if len(input_shape) == 1:
                shared = None
            else:
                shared = [i + 1 for i in range(len(input_shape) - 1)]
        return tf.keras.layers.PReLU(alpha_initializer=initializer, shared_axes=shared), input_shape
    elif framework == 'PyTorch':
        if share:
            shared = 1
        else:
            shared = input_shape[-1]
        return torch.nn.PReLU(init=alpha, num_parameters=shared), input_shape
    elif framework == 'Jittor':
        if share:
            shared = 1
        else:
            shared = input_shape[-1]
        return jittor.nn.PReLU(init_=alpha, num_parameters=shared), input_shape
    else:
        raise Exception('No support DL framework.')
