import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.slice import Slice as tf_slice
from src.configuration.support.torch_expansion.slice import Slice as torch_slice
from src.configuration.support.jittor_expansion.slice import Slice as jittor_slice


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    begin = params['begin']
    size = params['size']
    if len(begin) != len(size) or len(begin) != len(input_shape) + 1:
        raise Exception('Begin and Size of Slice layer should have the same ndim as the input shape.')
    tensor_space = params['tensor_space']
    # 计算output shape
    output_shape = list(input_shape)
    for i in range(1, len(size)):
        output_shape[i - 1] = size[i]
    if shape_only:
        return output_shape
    if framework == 'TensorFlow':
        return tf_slice(begin, size), output_shape
    elif framework == 'PyTorch':
        return torch_slice(begin, size, tensor_space), output_shape
    elif framework == 'Jittor':
        return jittor_slice(begin, size, tensor_space), output_shape
    else:
        raise Exception('No support DL framework.')
