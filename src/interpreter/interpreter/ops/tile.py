import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.tile import Tile as tf_tile
from src.configuration.support.torch_expansion.tile import Tile as torch_tile
from src.configuration.support.jittor_expansion.tile import Tile as jt_tile


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    multiples = params['multiples']
    if len(input_shape) != len(multiples):
        raise Exception('Length of \"multiples\" of layer Tile should be the same as input ndim.')
    tensor_space = params['tensor_space']
    output_shape = list(input_shape)
    for i in range(len(output_shape)):
        output_shape[i] *= multiples[i]
    if shape_only:
        return output_shape
    multiples = [1] + multiples
    if framework == 'TensorFlow':
        return tf_tile(multiples), output_shape
    elif framework == 'PyTorch':
        return torch_tile(multiples, tensor_space), output_shape
    elif framework == 'Jittor':
        return jt_tile(multiples, tensor_space), output_shape
    else:
        raise Exception('No support DL framework.')