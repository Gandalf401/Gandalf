import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.gather import Gather as tf_gather
from src.configuration.support.torch_expansion.gather import Gather as torch_gather
from src.configuration.support.jittor_expansion.gather import Gather as jittor_gather


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if params.__contains__('dim'):
        dim = params['dim']
    else:
        dim = 0
    index = params['index']
    tensor_space = params['tensor_space']
    # 计算输出尺寸
    output_shape = list(input_shape)
    if dim > 0:
        output_shape[dim - 1] = len(index)
    elif dim < 0:
        output_shape[dim] = len(index)
    if shape_only:
        return output_shape
    if framework == 'TensorFlow':
        return tf_gather(dim, index), output_shape
    elif framework == 'PyTorch':
        return torch_gather(dim, index, tensor_space), output_shape
    elif framework == 'Jittor':
        return jittor_gather(dim, index, tensor_space), output_shape
    else:
        raise Exception('No support DL framework.')
