import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.shape import Shape as tf_shape
from src.configuration.support.torch_expansion.shape import Shape as torch_shape
from src.configuration.support.jittor_expansion.shape import Shape as jittor_shape


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return [len(input_shape)]
    if framework == 'TensorFlow':
        return tf_shape(), [len(input_shape)]
    elif framework == 'PyTorch':
        return torch_shape(), [len(input_shape)]
    elif framework == 'Jittor':
        return jittor_shape(), [len(input_shape)]
    else:
        raise Exception('No support DL framework.')
