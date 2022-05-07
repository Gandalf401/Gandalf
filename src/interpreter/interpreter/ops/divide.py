import tensorflow as tf
import torch
import jittor

from src.configuration.support.torch_expansion.divide import Divide as torch_divide
from src.configuration.support.jittor_expansion.divide import Divide as jt_divide
from src.configuration.support.tf_expansion.divide import Divide as tf_divide

from src.interpreter.interpreter.ops._cal_broadcast_shape import cal_broadcast_shape


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    t1 = input_shape[0]
    t2 = input_shape[1]
    if not isinstance(t1, list) or not isinstance(t2, list):
        raise Exception('Input shape of layer Divide should be a list of two lists.')
    # 生成output_shape
    output_shape = cal_broadcast_shape(t1, t2)
    if shape_only:
        return output_shape
    if framework == 'TensorFlow':
        return tf_divide(), output_shape
    elif framework == 'PyTorch':
        return torch_divide(), output_shape
    elif framework == 'Jittor':
        return jt_divide(), output_shape
    else:
        raise Exception('No support DL framework.')
