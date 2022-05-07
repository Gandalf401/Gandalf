import math
import tensorflow as tf
import torch
import jittor

from src.interpreter.interpreter.ops.same_padding import cal_same_padding_1d
from src.configuration.support.torch_expansion.same_pad1d import SamePadding1d as t_same_padding
from src.configuration.support.jittor_expansion.same_pad1d import SamePadding1d as j_same_padding
from src.configuration.support.torch_expansion.causal_pad1d import CausalPadding1d as t_causal_padding
from src.configuration.support.jittor_expansion.causal_pad1d import CausalPadding1d as j_causal_padding


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    # 字段读取或默认初始化
    # padding
    if params.__contains__('padding'):
        padding = params['padding']
    else:
        padding = 'valid'
    # stride
    if params.__contains__('stride'):
        stride = params['stride']
    else:
        stride = 1
    # stride
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

    # in_ & out_channels
    in_channels = params['in_channels']
    out_channels = params['out_channels']
    # 檢查dilation和stride
    if dilation != 1 and stride != 1:
        raise Exception('Dilation and stride of Conv1d cannot be unequal to 1 at the same time.')

    if padding == 'same' or padding == 'causal':
        out_shape = [math.ceil(input_shape[0] / stride), out_channels]
    else:
        out_shape = [math.floor((input_shape[0] - dilation * (kernel_size - 1) - 1) / stride + 1), out_channels]
    if shape_only:
        return out_shape
    # 分框架实现
    if framework == 'TensorFlow':
        params_str = '(filters={0}, kernel_size={1}, strides={2}, padding=\"{3}\", use_bias={4}, dilation_rate={5})'\
            .format(out_channels, kernel_size, stride, padding, bias, dilation)
        return eval('tf.keras.layers.Conv1D' + params_str), out_shape
    elif framework == 'PyTorch':
        if padding == 'valid':
            params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding=0, bias={4}, ' \
                         'dilation={5})'.format(in_channels, out_channels, kernel_size, stride, bias, dilation)
            return eval('torch.nn.Conv1d' + params_str), out_shape
        else:
            pad = cal_same_padding_1d(input_shape, kernel_size, stride, dilation)
            if padding == 'same':
                if pad % 2 == 0:
                    params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding={4}, ' \
                                 'bias={5}, dilation={6})'.format(in_channels, out_channels, kernel_size,
                                                                  stride, pad // 2, bias, dilation)
                    return eval('torch.nn.Conv1d' + params_str), out_shape
                else:
                    pad_layer = t_same_padding(pad)
                    params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding={4}, ' \
                                 'bias={5}, dilation={6})'.format(in_channels, out_channels, kernel_size,
                                                    stride, pad // 2, bias, dilation)
                    return [pad_layer, eval('torch.nn.Conv1d' + params_str)], out_shape
            else:
                pad_layer = t_causal_padding(pad)
                params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding=0, bias={4}, ' \
                             'dilation={5})'.format(in_channels, out_channels, kernel_size, stride, bias, dilation)
                return [pad_layer, eval('torch.nn.Conv1d' + params_str)], out_shape
    elif framework == 'Jittor':
        if padding == 'valid':
            params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding=0, bias={4}, ' \
                         'dilation={5})'.format(in_channels, out_channels, kernel_size, stride, bias, dilation)
            return eval('jittor.nn.Conv1d' + params_str), out_shape
        else:
            pad = cal_same_padding_1d(input_shape, kernel_size, stride, dilation)
            if padding == 'same':
                if pad % 2 == 0:
                    params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding={4}, ' \
                                 'bias={5}, dilation={6})'.format(in_channels, out_channels, kernel_size,
                                                                  stride, pad // 2, bias, dilation)
                    return eval('jittor.nn.Conv1d' + params_str), out_shape
                else:
                    pad_layer = j_same_padding(pad)
                    params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding={4}, ' \
                                 'bias={5}, dilation={5})'.format(in_channels, out_channels, kernel_size,
                                                    stride, pad // 2, bias, dilation)
                    return [pad_layer, eval('jittor.nn.Conv1d' + params_str)], out_shape
            else:
                pad_layer = j_causal_padding(pad)
                params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding=0, bias={4}, ' \
                             'dilation={5})'.format(in_channels, out_channels, kernel_size, stride, bias, dilation)
                return [pad_layer, eval('jittor.nn.Conv1d' + params_str)], out_shape
    else:
        raise Exception('No support DL framework.')
