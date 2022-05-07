import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.top_k import TopK as tf_top_k
from src.configuration.support.torch_expansion.top_k import TopK as torch_top_k
from src.configuration.support.jittor_expansion.top_k import TopK as jittor_top_k


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    k = params['k']
    if params.__contains__('output'):
        output = params['output']
    else:
        output = 'indices'
    # 判断输出尺寸
    output_shape = list(input_shape)
    output_shape[-1] = k
    if shape_only:
        return output_shape
    # 返回算子和尺寸
    if framework == 'TensorFlow':
        return tf_top_k(k, output), output_shape
    elif framework == 'PyTorch':
        return torch_top_k(k, output), output_shape
    elif framework == 'Jittor':
        return jittor_top_k(k, output), output_shape
    else:
        raise Exception('No support DL framework.')
