import tensorflow as tf
import torch
import jittor

from src.configuration.support.torch_expansion.lamb import Lambda as torch_lamb
from src.configuration.support.jittor_expansion.lamb import Lambda as jittor_lamb


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    function = params['function']
    if isinstance(function, str):
        function = eval(function)
    output_shape = params['output_shape']
    if shape_only:
        return output_shape
    if framework == 'TensorFlow':
        return tf.keras.layers.Lambda(function), output_shape
    elif framework == 'PyTorch':
        return torch_lamb(function), output_shape
    elif framework == 'Jittor':
        return jittor_lamb(function), output_shape
    else:
        raise Exception('No support DL framework.')
