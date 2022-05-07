import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.argmin import Argmin as tf_min
from src.configuration.support.torch_expansion.argmin import Argmin as torch_min
from src.configuration.support.jittor_expansion.argmin import Argmin as jittor_min


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if params.__contains__('dim'):
        dim = params['dim']
    else:
        dim = 0
    if params.__contains__('channel_stay'):
        channel_stay = params['channel_stay']
    else:
        channel_stay = False
    # 判断输出尺寸
    output_shape = list(input_shape)
    if dim != 0:
        output_shape.pop(dim - 1)
    if shape_only:
        return output_shape
    # 返回算子和尺寸
    if framework == 'TensorFlow':
        return tf_min(dim), output_shape
    elif framework == 'PyTorch':
        return torch_min(dim, channel_stay), output_shape
    elif framework == 'Jittor':
        return jittor_min(dim, channel_stay), output_shape
    else:
        raise Exception('No support DL framework.')
