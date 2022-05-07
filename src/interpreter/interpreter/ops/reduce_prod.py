import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.reduce_prod import ReduceProd as tf_prod
from src.configuration.support.torch_expansion.reduce_prod import ReduceProd as torch_prod
from src.configuration.support.jittor_expansion.reduce_prod import ReduceProd as jittor_prod


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if params.__contains__('dim'):
        dim = params['dim']
    else:
        dim = None
    if params.__contains__('keep_dims'):
        keep_dims = params['keep_dims']
    else:
        keep_dims = False
    tensor_space = params['tensor_space']
    # 判断输出尺寸
    if keep_dims:
        if dim is None:
            output_shape = [1] * len(input_shape)
        else:
            output_shape = list(input_shape)
            if dim != 0:
                output_shape[dim - 1] = 1
    else:
        if dim is None:
            output_shape = []
        else:
            output_shape = list(input_shape)
            if dim != 0:
                output_shape.pop(dim - 1)
    if shape_only:
        return output_shape
    # 返回算子和尺寸
    if framework == 'TensorFlow':
        return tf_prod(dim, keep_dims), output_shape
    elif framework == 'PyTorch':
        return torch_prod(dim, keep_dims, tensor_space), output_shape
    elif framework == 'Jittor':
        return jittor_prod(dim, keep_dims, tensor_space), output_shape
    else:
        raise Exception('No support DL framework.')
