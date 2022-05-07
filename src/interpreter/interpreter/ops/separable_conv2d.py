import math
import tensorflow as tf
import torch
import jittor

from src.interpreter.interpreter.ops.same_padding import cal_same_padding
from src.configuration.support.torch_expansion.same_pad2d import SamePadding2d as t_same_padding
from src.configuration.support.jittor_expansion.same_pad2d import SamePadding2d as j_same_padding

from src.configuration.support.torch_expansion.separable_conv2d import SeparableConv2d as torch_conv2d
from src.configuration.support.jittor_expansion.separable_conv2d import SeparableConv2d as jittor_conv2d


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    # 字段读取或默认初始化
    # padding
    if params.__contains__('padding'):
        padding = params['padding']
    else:
        padding = 'valid'
    # depth_multiplier
    if params.__contains__('depth_multiplier'):
        depth_multiplier = params['depth_multiplier']
    else:
        depth_multiplier = 1
    # stride
    if params.__contains__('stride'):
        if isinstance(params['stride'], int):
            stride = (params['stride'], params['stride'])
        else:
            stride = tuple(params['stride'])
    else:
        stride = (1, 1)
    # dilation
    if params.__contains__('dilation'):
        if isinstance(params['dilation'], int):
            dilation = (params['dilation'], params['dilation'])
        else:
            dilation = tuple(params['dilation'])
    else:
        dilation = (1, 1)
    # kernel_size
    if isinstance(params['kernel_size'], int):
        kernel_size = (params['kernel_size'], params['kernel_size'])
    else:
        kernel_size = tuple(params['kernel_size'])
    # bias
    if params.__contains__('bias'):
        bias = params['bias']
    else:
        bias = True
    # 检查
    if dilation[0] != 1 and stride[0] != 1 or dilation[1] != 1 and stride[1] != 1:
        raise Exception('Dilation and stride of SeparableConv2d cannot be unequal to 1 at the same time.')

    # in_ & out_channels
    in_channels = params['in_channels']
    out_channels = params['out_channels']

    if padding == 'same':
        out_shape = [math.ceil(input_shape[0] / stride[0]), math.ceil(input_shape[1] / stride[1]), out_channels]
    else:
        out_shape = [math.floor((input_shape[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1),
                     math.floor((input_shape[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1),
                     out_channels]
    if shape_only:
        return out_shape
    # 分框架实现
    if framework == 'TensorFlow':
        params_str = '(kernel_size={0}, strides={1}, padding=\"{2}\", depth_multiplier={3}, dilation_rate={4}, ' \
                     'use_bias={5}, filters={6})'.format(kernel_size, stride, padding, depth_multiplier,
                                                         dilation, bias, out_channels)
        return eval('tf.keras.layers.SeparableConv2D' + params_str), out_shape
    elif framework == 'PyTorch':
        if padding == 'valid':
            params_str = '(in_channels={0}, kernel_size={1}, stride={2}, padding=0, dilation={3}, ' \
                         'depth_multiplier={4}, bias={5}, out_channels={6})'.format(
                in_channels, kernel_size, stride, dilation, depth_multiplier, bias, out_channels)
            return eval('torch_conv2d' + params_str), out_shape
        else:
            pad_h, pad_w = cal_same_padding(input_shape, kernel_size, stride, dilation)
            if pad_h % 2 == 0 and pad_w % 2 == 0:
                params_str = '(in_channels={0}, kernel_size={1}, stride={2}, padding=({3}, {4}), dilation={5}, ' \
                             'bias={6}, out_channels={7}, depth_multiplier={8})'.format(in_channels, kernel_size,
                                                                                        stride, pad_h // 2, pad_w // 2,
                                                                                        dilation, bias, out_channels,
                                                                                        depth_multiplier)
                return eval('torch_conv2d' + params_str), out_shape
            else:
                pad = t_same_padding(pad_h, pad_w)
                params_str = '(in_channels={0}, kernel_size={1}, stride={2}, padding=({3}, {4}), dilation={5}, ' \
                             'bias={6}, out_channels={7}, depth_multiplier={8})'.format(in_channels, kernel_size,
                                                                                        stride, pad_h // 2, pad_w // 2,
                                                                                        dilation, bias, out_channels,
                                                                                        depth_multiplier)
                return [pad, eval('torch_conv2d' + params_str)], out_shape
    elif framework == 'Jittor':
        if padding == 'valid':
            params_str = '(in_channels={0}, kernel_size={1}, stride={2}, padding=0, dilation={3}, ' \
                         'depth_multiplier={4}, bias={5}, out_channels={6})'.format(
                in_channels, kernel_size, stride, dilation, depth_multiplier, bias, out_channels)
            return eval('jittor_conv2d' + params_str), out_shape
        else:
            pad_h, pad_w = cal_same_padding(input_shape, kernel_size, stride, dilation)
            if pad_h % 2 == 0 and pad_w % 2 == 0:
                params_str = '(in_channels={0}, kernel_size={1}, stride={2}, padding=({3}, {4}), dilation={5}, ' \
                             'bias={6}, out_channels={7}, depth_multiplier={8})'.format(in_channels, kernel_size,
                                                                                        stride, pad_h // 2, pad_w // 2,
                                                                                        dilation, bias, out_channels,
                                                                                        depth_multiplier)
                return eval('jittor_conv2d' + params_str), out_shape
            else:
                pad = j_same_padding(pad_h, pad_w)
                params_str = '(in_channels={0}, kernel_size={1}, stride={2}, padding=({3}, {4}), dilation={5}, ' \
                             'bias={6}, out_channels={7}, depth_multiplier={8})'.format(in_channels, kernel_size,
                                                                                        stride, pad_h // 2, pad_w // 2,
                                                                                        dilation, bias, out_channels,
                                                                                        depth_multiplier)
                return [pad, eval('jittor_conv2d' + params_str)], out_shape
    else:
        raise Exception('No support DL framework.')
