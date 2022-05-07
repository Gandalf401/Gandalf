import math

import tensorflow as tf
import torch
import jittor

from src.interpreter.interpreter.ops.same_padding import cal_same_padding_3d
from src.configuration.support.torch_expansion.same_pad3d import SamePadding3d as t_same_padding
from src.configuration.support.jittor_expansion.same_pad3d import SamePadding3d as j_same_padding


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if isinstance(params['pool_size'], int):
        pool_size = (params['pool_size'], params['pool_size'], params['pool_size'])
    else:
        pool_size = tuple(params['pool_size'])
    if params.__contains__('stride'):
        if isinstance(params['stride'], int):
            stride = (params['stride'], params['stride'], params['stride'])
        else:
            stride = tuple(params['stride'])
    else:
        stride = (1, 1)
    if params.__contains__('padding'):
        padding = params['padding']
    else:
        padding = 'valid'
    # 输出维度
    if padding == 'same':
        out_shape = [math.ceil(input_shape[0] / stride[0]), math.ceil(input_shape[1] / stride[1]),
                     math.ceil(input_shape[2] / stride[2]), input_shape[-1]]
    else:
        out_shape = [math.ceil((input_shape[0] - pool_size[0] + 1) / stride[0]),
                     math.ceil((input_shape[1] - pool_size[1] + 1) / stride[1]),
                     math.ceil((input_shape[2] - pool_size[2] + 1) / stride[2]),
                     input_shape[-1]]
    if shape_only:
        return out_shape
    if framework == 'TensorFlow':
        return tf.keras.layers.AveragePooling3D(pool_size, stride, padding), out_shape
    elif framework == 'PyTorch':
        if padding == 'valid':
            params_str = '(kernel_size={0}, stride={1}, padding=0)'.format(pool_size, stride)
            return eval('torch.nn.AvgPool3d' + params_str), out_shape
        else:
            pad_d, pad_h, pad_w = cal_same_padding_3d(input_shape, pool_size, stride)
            if pad_d % 2 == 0 and pad_h % 2 == 0 and pad_w % 2 == 0:
                params_str = '(kernel_size={0}, stride={1}, padding=({2}, {3}, {4}))' \
                    .format(pool_size, stride, pad_d // 2, pad_h // 2, pad_w // 2)
                return eval('torch.nn.AvgPool3d' + params_str), out_shape
            else:
                pad = t_same_padding(pad_d, pad_h, pad_w)
                params_str = '(kernel_size={0}, stride={1}, padding=({2}, {3}, {4}))' \
                    .format(pool_size, stride, pad_d // 2, pad_h // 2, pad_w // 2)
                return [pad, eval('torch.nn.AvgPool3d' + params_str)], out_shape
    elif framework == 'Jittor':
        if padding == 'valid':
            params_str = '(kernel_size={0}, stride={1}, padding=0)'.format(pool_size, stride)
            return eval('jittor.nn.AvgPool3d' + params_str), out_shape
        else:
            pad_d, pad_h, pad_w = cal_same_padding_3d(input_shape, pool_size, stride)
            if pad_d % 2 == 0 and pad_h % 2 == 0 and pad_w % 2 == 0:
                params_str = '(kernel_size={0}, stride={1}, padding=({2}, {3}, {4}))' \
                    .format(pool_size, stride, pad_d // 2, pad_h // 2, pad_w // 2)
                return eval('jittor.nn.AvgPool3d' + params_str), out_shape
            else:
                pad = j_same_padding(pad_d, pad_h, pad_w)
                params_str = '(kernel_size={0}, stride={1}, padding=({2}, {3}, {4}))' \
                    .format(pool_size, stride, pad_d // 2, pad_h // 2, pad_w // 2)
                return [pad, eval('jittor.nn.AvgPool3d' + params_str)], out_shape
    else:
        raise Exception('No support DL framework.')
