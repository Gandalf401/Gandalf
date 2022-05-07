import tensorflow as tf
import torch
import jittor

from src.configuration.support.torch_expansion.subtract import Subtract as t_subtract
from src.configuration.support.jittor_expansion.subtract import Subtract as jt_subtract

from src.interpreter.interpreter.ops._cal_broadcast_shape import cal_broadcast_shape


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    t1 = input_shape[0]
    t2 = input_shape[1]
    if not isinstance(t1, list) or not isinstance(t2, list):
        raise Exception('Input shape of layer Subtract should be a list of two lists.')
    # if t1[0] != t2[0]:
    #     raise Exception('Input tensors of layer Subtract should be with the same batch size.')
    # 生成output_shape
    output_shape = cal_broadcast_shape(t1, t2)
    if shape_only:
        return output_shape
    if framework == 'TensorFlow':
        return tf.keras.layers.Subtract(), output_shape
    elif framework == 'PyTorch':
        return t_subtract(), output_shape
    elif framework == 'Jittor':
        return jt_subtract(), output_shape
    else:
        raise Exception('No support DL framework.')
