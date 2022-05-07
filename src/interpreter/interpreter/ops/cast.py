import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.cast import Cast as tf_cast
from src.configuration.support.torch_expansion.cast import Cast as torch_cast
from src.configuration.support.jittor_expansion.cast import Cast as jittor_cast


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    target_dtype = params['target_dtype']
    if shape_only:
        return input_shape
    # 返回算子和尺寸
    if framework == 'TensorFlow':
        return tf_cast(target_dtype), input_shape
    elif framework == 'PyTorch':
        return torch_cast(target_dtype), input_shape
    elif framework == 'Jittor':
        return jittor_cast(target_dtype), input_shape
    else:
        raise Exception('No support DL framework.')
