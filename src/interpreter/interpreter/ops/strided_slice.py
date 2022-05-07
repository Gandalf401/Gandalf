import math

import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.strided_slice import StridedSlice as tf_strided_slice
from src.configuration.support.torch_expansion.strided_slice import StridedSlice as torch_strided_slice
from src.configuration.support.jittor_expansion.strided_slice import StridedSlice as jittor_strided_slice


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    begin = params['begin']
    end = params['end']
    if params.__contains__('stride'):
        stride = params['stride']
    else:
        stride = [1 for _ in range(len(begin))]
    if len(begin) != len(end) != len(stride) or len(begin) != len(input_shape) + 1:
        raise Exception('Begin, End and Stride of StridedSlice layer should have the same ndim as the input shape.')
    tensor_space = params['tensor_space']
    # 计算output shape
    output_shape = list(input_shape)
    for i in range(1, len(begin)):
        if (end[i] - begin[i]) % stride[i] == 0:
            output_shape[i - 1] = math.floor((end[i] - begin[i]) / stride[i])
        else:
            output_shape[i - 1] = math.floor((end[i] - begin[i]) / stride[i]) + 1
    if shape_only:
        return output_shape
    if framework == 'TensorFlow':
        return tf_strided_slice(begin, end, stride), output_shape
    elif framework == 'PyTorch':
        return torch_strided_slice(begin, end, stride, tensor_space), output_shape
    elif framework == 'Jittor':
        return jittor_strided_slice(begin, end, stride, tensor_space), output_shape
    else:
        raise Exception('No support DL framework.')
