import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.compare import Compare as tf_compare
from src.configuration.support.torch_expansion.compare import Compare as torch_compare
from src.configuration.support.jittor_expansion.compare import Compare as jittor_compare
from src.interpreter.interpreter.ops._cal_broadcast_shape import cal_broadcast_shape


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    op = params['op']
    t1 = input_shape[0]
    t2 = input_shape[1]
    if not isinstance(t1, list) or not isinstance(t2, list):
        raise Exception('Input shape of layer Compare should be a list of two lists.')
    output_shape = cal_broadcast_shape(t1, t2)
    if shape_only:
        return output_shape
    # 返回算子和尺寸
    if framework == 'TensorFlow':
        return tf_compare(op), output_shape
    elif framework == 'PyTorch':
        return torch_compare(op), output_shape
    elif framework == 'Jittor':
        return jittor_compare(op), output_shape
    else:
        raise Exception('No support DL framework.')
