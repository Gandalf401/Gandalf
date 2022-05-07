import tensorflow as tf
import torch
import jittor

from src.configuration.support.torch_expansion.flatten import Flatten as torch_flatten
from src.configuration.support.jittor_expansion.flatten import Flatten as jt_flatten


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    sum = 1
    for x in input_shape:
        sum *= x
    if shape_only:
        return [sum]
    if framework == 'TensorFlow':
        return tf.keras.layers.Flatten(), [sum]
    elif framework == 'PyTorch':
        return torch_flatten(), [sum]
    elif framework == 'Jittor':
        return jt_flatten(), [sum]
    else:
        raise Exception('No support DL framework.')