import math
import tensorflow as tf
import torch
import jittor

from src.interpreter.interpreter.ops.same_padding import cal_same_padding_1d
from src.configuration.support.torch_expansion.same_pad1d import SamePadding1d as t_same_padding
from src.configuration.support.jittor_expansion.same_pad1d import SamePadding1d as j_same_padding

from src.configuration.support.torch_expansion.depthwise_conv1d import DepthwiseConv1d as torch_conv1d
from src.configuration.support.jittor_expansion.depthwise_conv1d import DepthwiseConv1d as jittor_conv1d


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
        stride = params['stride']
    else:
        stride = 1
    # dilation
    if params.__contains__('dilation'):
        dilation = params['dilation']
    else:
        dilation = 1
    # kernel_size
    kernel_size = params['kernel_size']
    # bias
    if params.__contains__('bias'):
        bias = params['bias']
    else:
        bias = True
    # 检查
    if dilation != 1 and stride != 1:
        raise Exception('Dilation and stride of DepthwiseConv1d cannot be unequal to 1 at the same time.')

    # in_ & out_channels
    in_channels = params['in_channels']

    if padding == 'same':
        out_shape = [math.ceil(input_shape[0] / stride), in_channels * depth_multiplier]
    else:
        out_shape = [math.floor((input_shape[0] - dilation * (kernel_size - 1) - 1) / stride + 1),
                     in_channels * depth_multiplier]

    if shape_only:
        return out_shape
    # 分框架实现
    if framework == 'TensorFlow':
        params_str = '(kernel_size={0}, strides={1}, padding=\"{2}\", depth_multiplier={3}, dilation_rate={4}, ' \
                     'use_bias={5})'.format(kernel_size, stride, padding, depth_multiplier, dilation, bias)
        return eval('tf.keras.layers.DepthwiseConv1D' + params_str), out_shape
    elif framework == 'PyTorch':
        if padding == 'valid':
            params_str = '(in_channels={0}, kernel_size={1}, stride={2}, padding=0, dilation={3}, ' \
                         'depth_multiplier={4}, bias={5})'.format(
                in_channels, kernel_size, stride, dilation, depth_multiplier, bias)
            return eval('torch_conv1d' + params_str), out_shape
        else:
            pad = cal_same_padding_1d(input_shape, kernel_size, stride, dilation)
            if pad % 2 == 0:
                params_str = '(in_channels={0}, kernel_size={1}, stride={2}, padding={3}, dilation={4}, ' \
                             'bias={5}, depth_multiplier={6})'.format(in_channels, kernel_size, stride,
                                                pad // 2, dilation, bias, depth_multiplier)
                return eval('torch_conv1d' + params_str), out_shape
            else:
                pad_layer = t_same_padding(pad)
                params_str = '(in_channels={0}, kernel_size={1}, stride={2}, padding={3}, dilation={4}, ' \
                             'bias={5}, depth_multiplier={6})'.format(in_channels, kernel_size, stride,
                                                pad // 2, dilation, bias, depth_multiplier)
                return [pad_layer, eval('torch_conv1d' + params_str)], out_shape
    elif framework == 'Jittor':
        if padding == 'valid':
            params_str = '(in_channels={0}, kernel_size={1}, stride={2}, padding=0, dilation={3}, ' \
                         'depth_multiplier={4}, bias={5})'.format(
                in_channels, kernel_size, stride, dilation, depth_multiplier, bias)
            return eval('jittor_conv1d' + params_str), out_shape
        else:
            pad = cal_same_padding_1d(input_shape, kernel_size, stride, dilation)
            if pad % 2 == 0:
                params_str = '(in_channels={0}, kernel_size={1}, stride={2}, padding={3}, dilation={4}, ' \
                             'bias={5}, depth_multiplier={6})'.format(in_channels, kernel_size, stride,
                                                pad // 2, dilation, bias, depth_multiplier)
                return eval('jittor_conv1d' + params_str), out_shape
            else:
                pad_layer = j_same_padding(pad)
                params_str = '(in_channels={0}, kernel_size={1}, stride={2}, padding={3}, dilation={4}, ' \
                             'bias={5}, depth_multiplier={6})'.format(in_channels, kernel_size, stride,
                                                pad // 2, dilation, bias, depth_multiplier)
                return [pad_layer, eval('jittor_conv1d' + params_str)], out_shape
    else:
        raise Exception('No support DL framework.')
