import math
import tensorflow as tf
import torch
import jittor
from src.interpreter.interpreter.ops.same_padding import cal_same_padding
from src.configuration.support.torch_expansion.same_pad2d import SamePadding2d as t_same_padding
from src.configuration.support.jittor_expansion.same_pad2d import SamePadding2d as j_same_padding


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    # 字段读取或默认初始化
    # padding
    if params.__contains__('padding'):
        padding = params['padding']
    else:
        padding = 'valid'
    # stride
    if params.__contains__('stride'):
        if isinstance(params['stride'], int):
            stride = (params['stride'], params['stride'])
        else:
            stride = tuple(params['stride'])
    else:
        stride = (1, 1)
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

    # in_ & out_channels
    in_channels = params['in_channels']
    out_channels = params['out_channels']
    # dilation
    if params.__contains__('dilation'):
        if isinstance(params['dilation'], int):
            dilation = (params['dilation'], params['dilation'])
        else:
            dilation = tuple(params['dilation'])
    else:
        dilation = (1, 1)
    if padding == 'same':
        out_shape = [math.ceil(input_shape[0] / stride[0]), math.ceil(input_shape[1] / stride[1]), out_channels]
    else:
        out_shape = [math.floor((input_shape[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1),
                     math.floor((input_shape[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1),
                     out_channels]
    # 檢查dilation和stride
    if dilation[0] != 1 and stride[0] != 1 or dilation[1] != 1 and stride[1] != 1:
        raise Exception('Dilation and stride of Conv2d cannot be unequal to 1 at the same time.')
    if shape_only:
        return out_shape
    # 分框架实现
    if framework == 'TensorFlow':
        params_str = '(filters={0}, kernel_size={1}, strides={2}, padding=\"{3}\", use_bias={4}, dilation_rate={5})'\
            .format(out_channels, kernel_size, stride, padding, bias, dilation)
        return eval('tf.keras.layers.Conv2D' + params_str), out_shape
    elif framework == 'PyTorch':
        if padding == 'valid':
            params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding=0, bias={4}, ' \
                         'dilation={5})'.format(in_channels, out_channels, kernel_size, stride, bias, dilation)
            return eval('torch.nn.Conv2d' + params_str), out_shape
        else:
            pad_h, pad_w = cal_same_padding(input_shape, kernel_size, stride, dilation)
            if pad_h % 2 == 0 and pad_w % 2 == 0:
                params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding=({4}, {5}), ' \
                             'bias={6}, dilation={7})'.format(in_channels, out_channels, kernel_size,
                                                stride, pad_h // 2, pad_w // 2, bias, dilation)
                return eval('torch.nn.Conv2d' + params_str), out_shape
            else:
                pad = t_same_padding(pad_h, pad_w)
                params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding=({4}, {5}), ' \
                             'bias={6}, dilation={7})'.format(in_channels, out_channels, kernel_size,
                                                stride, pad_h // 2, pad_w // 2, bias, dilation)
                return [pad, eval('torch.nn.Conv2d' + params_str)], out_shape
    elif framework == 'Jittor':
        if padding == 'valid':
            params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding=0, bias={4}, ' \
                         'dilation={5})'.format(in_channels, out_channels, kernel_size, stride, bias, dilation)
            return eval('jittor.nn.Conv2d' + params_str), out_shape
        else:
            pad_h, pad_w = cal_same_padding(input_shape, kernel_size, stride, dilation)
            if pad_h % 2 == 0 and pad_w % 2 == 0:
                params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding=({4}, {5}), ' \
                             'bias={6}, dilation={7})'.format(in_channels, out_channels, kernel_size,
                                               stride, pad_h // 2, pad_w // 2, bias, dilation)
                return eval('jittor.nn.Conv2d' + params_str), out_shape
            else:
                pad = j_same_padding(pad_h, pad_w)
                params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding=({4}, {5}), ' \
                             'bias={6}, dilation={7})'.format(in_channels, out_channels, kernel_size,
                                                stride, pad_h // 2, pad_w // 2, bias, dilation)
                return [pad, eval('jittor.nn.Conv2d' + params_str)], out_shape
    else:
        raise Exception('No support DL framework.')
