import math

import tensorflow as tf
import torch
import jittor

from src.interpreter.interpreter.ops.same_padding import cal_same_padding_1d
from src.configuration.support.torch_expansion.same_pad1d import SamePadding1d as t_same_padding
from src.configuration.support.jittor_expansion.same_pad1d import SamePadding1d as j_same_padding
from src.configuration.support.jittor_expansion.max_pool1d import MaxPool1d


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    pool_size = params['pool_size']
    stride = params['stride']
    if params.__contains__('padding'):
        padding = params['padding']
    else:
        padding = 'valid'
    # 输出维度
    if padding == 'same':
        out_shape = [math.ceil(input_shape[0] / stride), input_shape[-1]]
    else:
        out_shape = [math.ceil((input_shape[0] - pool_size + 1) / stride), input_shape[-1]]
    if shape_only:
        return out_shape
    if framework == 'TensorFlow':
        return tf.keras.layers.MaxPooling1D(pool_size, stride, padding), out_shape
    elif framework == 'PyTorch':
        if padding == 'valid':
            params_str = '(kernel_size={0}, stride={1}, padding=0)'.format(pool_size, stride)
            return eval('torch.nn.MaxPool1d' + params_str), out_shape
        else:
            pad = cal_same_padding_1d(input_shape, pool_size, stride)
            if pad % 2 == 0:
                params_str = '(kernel_size={0}, stride={1}, padding={2})'.format(pool_size, stride, pad // 2)
                return eval('torch.nn.MaxPool1d' + params_str), out_shape
            else:
                pad_layer = t_same_padding(pad)
                params_str = '(kernel_size={0}, stride={1}, padding={2})'.format(pool_size, stride, pad // 2)
                return [pad_layer, eval('torch.nn.MaxPool1d' + params_str)], out_shape
    elif framework == 'Jittor':
        if padding == 'valid':
            params_str = '(pool_size={0}, stride={1}, padding=0)'.format(pool_size, stride)
            return eval('MaxPool1d' + params_str), out_shape
        else:
            pad = cal_same_padding_1d(input_shape, pool_size, stride)
            if pad % 2 == 0:
                params_str = '(pool_size={0}, stride={1}, padding={2})'.format(pool_size, stride, pad // 2)
                return eval('MaxPool1d' + params_str), out_shape
            else:
                pad_layer = j_same_padding(pad)
                params_str = '(pool_size={0}, stride={1}, padding={2})'.format(pool_size, stride, pad // 2)
                return [pad_layer, eval('MaxPool1d' + params_str)], out_shape
    else:
        raise Exception('No support DL framework.')
