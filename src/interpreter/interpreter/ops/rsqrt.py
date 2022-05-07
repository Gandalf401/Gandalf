import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.rsqrt import Rsqrt as tf_rsqrt
from src.configuration.support.torch_expansion.rsqrt import Rsqrt as torch_rsqrt
from src.configuration.support.jittor_expansion.rsqrt import Rsqrt as jittor_rsqrt


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return input_shape
    if framework == 'TensorFlow':
        return tf_rsqrt(), input_shape
    elif framework == 'PyTorch':
        return torch_rsqrt(), input_shape
    elif framework == 'Jittor':
        return jittor_rsqrt(), input_shape
    else:
        raise Exception('No support DL framework.')
