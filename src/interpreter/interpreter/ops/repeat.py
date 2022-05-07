import numpy as np
import tensorflow as tf
import torch
import jittor

from src.configuration.support.torch_expansion.repeat import Repeat as torch_repeat
from src.configuration.support.jittor_expansion.repeat import Repeat as jittor_repeat


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if len(input_shape) != 1:
        raise Exception('Only support repeat operations for vector.')
    n = params['n']
    output_shape = list(input_shape)
    output_shape.insert(0, n)
    if shape_only:
        return output_shape
    if framework == 'TensorFlow':
        return tf.keras.layers.RepeatVector(n), output_shape
    elif framework == 'PyTorch':
        return torch_repeat(n), output_shape
    elif framework == 'Jittor':
        return jittor_repeat(n), output_shape
    else:
        raise Exception('No support DL framework.')
