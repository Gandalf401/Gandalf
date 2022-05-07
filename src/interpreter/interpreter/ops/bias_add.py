import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.bias_add import BiasAdd as tf_add
from src.configuration.support.torch_expansion.bias_add import BiasAdd as torch_add
from src.configuration.support.jittor_expansion.bias_add import BiasAdd as jittor_add


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    bias = params['bias']
    if shape_only:
        return input_shape
    if framework == 'TensorFlow':
        return tf_add(bias), input_shape
    elif framework == 'PyTorch':
        return torch_add(bias), input_shape
    elif framework == 'Jittor':
        return jittor_add(bias), input_shape
    else:
        raise Exception('No support DL framework.')
