import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.floor import Floor as tf_floor
from src.configuration.support.torch_expansion.floor import Floor as torch_floor
from src.configuration.support.jittor_expansion.floor import Floor as jittor_floor


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return input_shape
    if framework == 'TensorFlow':
        return tf_floor(), input_shape
    elif framework == 'PyTorch':
        return torch_floor(), input_shape
    elif framework == 'Jittor':
        return jittor_floor(), input_shape
    else:
        raise Exception('No support DL framework.')
