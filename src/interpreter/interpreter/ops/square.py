import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.square import Square as tf_square
from src.configuration.support.torch_expansion.square import Square as torch_square
from src.configuration.support.jittor_expansion.square import Square as jittor_square


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return input_shape
    if framework == 'TensorFlow':
        return tf_square(), input_shape
    elif framework == 'PyTorch':
        return torch_square(), input_shape
    elif framework == 'Jittor':
        return jittor_square(), input_shape
    else:
        raise Exception('No support DL framework.')
