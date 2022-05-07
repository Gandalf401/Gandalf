import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.selu import SeLU as tf_selu
from src.configuration.support.jittor_expansion.selu import SeLU as jt_selu


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return input_shape
    if framework == 'TensorFlow':
        return tf_selu(), input_shape
    elif framework == 'PyTorch':
        return torch.nn.SELU(), input_shape
    elif framework == 'Jittor':
        return jt_selu(), input_shape
    else:
        raise Exception('No support DL framework.')
