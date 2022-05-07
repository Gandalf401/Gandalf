import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.transpose import Transpose as tf_transpose
from src.configuration.support.torch_expansion.transpose import Transpose as torch_transpose
from src.configuration.support.jittor_expansion.transpose import Transpose as jittor_transpose


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    output_shape = params['output_shape']
    tensor_space = params['tensor_space']
    if output_shape[0] != 0:
        raise Exception('Do not support transpose with axis 0.')
    output_shape_removed_zero = list(output_shape)
    output_shape_removed_zero = output_shape_removed_zero[1:]
    output_shape_lst = list(map(lambda x: input_shape[x - 1], output_shape_removed_zero))
    if shape_only:
        return output_shape_lst
    if framework == 'TensorFlow':
        return tf_transpose(output_shape), output_shape_lst
    elif framework == 'PyTorch':
        return torch_transpose(output_shape, tensor_space), output_shape_lst
    elif framework == 'Jittor':
        return jittor_transpose(output_shape, tensor_space), output_shape_lst
    else:
        raise Exception('No support DL framework.')
