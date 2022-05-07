import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.exp import Exp as tf_exp
from src.configuration.support.torch_expansion.exp import Exp as torch_exp
from src.configuration.support.jittor_expansion.exp import Exp as jittor_exp


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return input_shape
    if framework == 'TensorFlow':
        return tf_exp(), input_shape
    elif framework == 'PyTorch':
        return torch_exp(), input_shape
    elif framework == 'Jittor':
        return jittor_exp(), input_shape
    else:
        raise Exception('No support DL framework.')
