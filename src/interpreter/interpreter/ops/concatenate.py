import tensorflow as tf
import torch
import jittor

from src.configuration.support.torch_expansion.concatenate import Concatenate as t_cat
from src.configuration.support.jittor_expansion.concatenate import Concatenate as jt_cat


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if params.__contains__('dim'):
        dim = params['dim']
    else:
        dim = -1
    if not isinstance(input_shape[0], list) or not isinstance(input_shape[1], list):
        raise Exception('Input shape of layer Concatenate should be a list of two lists.')
    # 计算生成的tensor的尺寸维度
    output_shape = list(input_shape)[0]
    for i in range(1, len(input_shape)):
        output_shape[dim] += input_shape[i][dim]
    if shape_only:
        return output_shape
    if framework == 'TensorFlow':
        return tf.keras.layers.Concatenate(axis=dim), output_shape
    elif framework == 'PyTorch':
        return t_cat(axis=dim), output_shape
    elif framework == 'Jittor':
        return jt_cat(axis=dim), output_shape
    else:
        raise Exception('No support DL framework.')
