import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.squeeze import Squeeze as tf_squeeze
from src.configuration.support.torch_expansion.squeeze import Squeeze as torch_squeeze
from src.configuration.support.jittor_expansion.squeeze import Squeeze as jittor_squeeze


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    dim = params['dim']
    tensor_space = params['tensor_space']
    output_shape = list(input_shape)
    if dim == -1 or dim == len(input_shape):
        if input_shape[-1] == 1:
            output_shape.pop()
        else:
            raise Exception('The dim to be squeezed has more than one data.')
    elif dim != 0 and dim != -len(input_shape) - 1:
        if dim > 0:
            if input_shape[dim - 1] == 1:
                output_shape.pop(dim - 1)
            else:
                raise Exception('The dim to be squeezed has more than one data.')
        else:
            if input_shape[dim] == 1:
                output_shape.pop(dim)
            else:
                raise Exception('The dim to be squeezed has more than one data.')
    if shape_only:
        return output_shape
    if framework == 'TensorFlow':
        return tf_squeeze(dim), output_shape
    elif framework == 'PyTorch':
        return torch_squeeze(dim, tensor_space), output_shape
    elif framework == 'Jittor':
        return jittor_squeeze(dim, tensor_space), output_shape
    else:
        raise Exception('No support DL framework.')
