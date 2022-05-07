import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.resize import Resize as tf_resize
from src.configuration.support.torch_expansion.resize import Resize as torch_resize


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    output_shape = params['output_shape']
    if isinstance(output_shape, int):
        output_shape = (output_shape, output_shape)
    elif isinstance(output_shape, list):
        output_shape = (output_shape[0], output_shape[1])
    if params.__contains__('mode'):
        mode = params['mode']
    else:
        mode = 'nearest'

    output_shape_lst = [output_shape[0], output_shape[1], input_shape[-1]]

    if shape_only:
        return output_shape_lst
    if framework == 'TensorFlow':
        return tf_resize(output_shape, mode), output_shape_lst
    elif framework == 'PyTorch':
        return torch_resize(output_shape, mode), output_shape_lst
    elif framework == 'Jittor':
        return jittor.nn.Resize(output_shape, mode), output_shape_lst
    else:
        raise Exception('No support DL framework.')
