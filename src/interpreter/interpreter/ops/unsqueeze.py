import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.unsqueeze import Unsqueeze as tf_unsqueeze
from src.configuration.support.torch_expansion.unsqueeze import Unsqueeze as torch_unsqueeze
from src.configuration.support.jittor_expansion.unsqueeze import Unsqueeze as jittor_unsqueeze


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    dim = params['dim']
    tensor_space = params['tensor_space']
    output_shape = list(input_shape)
    if dim > 0:
        output_shape.insert(dim - 1, 1)
    elif dim < 0:
        output_shape.insert(dim, 1)
    if shape_only:
        return output_shape
    if framework == 'TensorFlow':
        return tf_unsqueeze(dim), output_shape
    elif framework == 'PyTorch':
        return torch_unsqueeze(dim, tensor_space), output_shape
    elif framework == 'Jittor':
        return jittor_unsqueeze(dim, tensor_space), output_shape
    else:
        raise Exception('No support DL framework.')
