import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.ceil import Ceil as tf_ceil
from src.configuration.support.torch_expansion.ceil import Ceil as torch_ceil
from src.configuration.support.jittor_expansion.ceil import Ceil as jittor_ceil


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return input_shape
    if framework == 'TensorFlow':
        return tf_ceil(), input_shape
    elif framework == 'PyTorch':
        return torch_ceil(), input_shape
    elif framework == 'Jittor':
        return jittor_ceil(), input_shape
    else:
        raise Exception('No support DL framework.')

