import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.sqrt import Sqrt as tf_sqrt
from src.configuration.support.torch_expansion.sqrt import Sqrt as torch_sqrt
from src.configuration.support.jittor_expansion.sqrt import Sqrt as jittor_sqrt


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return input_shape
    if framework == 'TensorFlow':
        return tf_sqrt(), input_shape
    elif framework == 'PyTorch':
        return torch_sqrt(), input_shape
    elif framework == 'Jittor':
        return jittor_sqrt(), input_shape
    else:
        raise Exception('No support DL framework.')
