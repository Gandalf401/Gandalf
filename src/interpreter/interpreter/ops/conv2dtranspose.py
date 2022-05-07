import numpy as np
import tensorflow as tf
import torch
import jittor
from src.interpreter.interpreter.ops.same_padding import cal_same_padding


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    # 字段读取或默认初始化
    # padding
    if params.__contains__('padding'):
        padding = params['padding']
    else:
        padding = 'valid'
    if params.__contains__('out_padding'):
        if isinstance(params['out_padding'], int):
            out_padding = (params['out_padding'], params['out_padding'])
        else:
            out_padding = tuple(params['out_padding'])
    else:
        out_padding = (0, 0)
    # stride
    if params.__contains__('stride'):
        if isinstance(params['stride'], int):
            stride = (params['stride'], params['stride'])
        else:
            stride = tuple(params['stride'])
    else:
        stride = (1, 1)
    # bias
    if params.__contains__('bias'):
        bias = params['bias']
    else:
        bias = True
    # kernel_size
    if isinstance(params['kernel_size'], int):
        kernel_size = (params['kernel_size'], params['kernel_size'])
    else:
        kernel_size = tuple(params['kernel_size'])
    # dilation
    if params.__contains__('dilation'):
        if isinstance(params['dilation'], int):
            dilation = (params['dilation'], params['dilation'])
        else:
            dilation = tuple(params['dilation'])
    else:
        dilation = (1, 1)
    # in_ & out_channels
    in_channels = params['in_channels']
    out_channels = params['out_channels']
    if padding == 'same':
        out_shape = [input_shape[0] * stride[0] + out_padding[0], input_shape[1] * stride[1] + out_padding[1],
                     out_channels]
    else:
        out_shape = [(input_shape[0] - 1) * stride[0] + dilation[0] * (kernel_size[0] - 1) + out_padding[0] + 1,
                     (input_shape[1] - 1) * stride[1] + dilation[1] * (kernel_size[1] - 1) + out_padding[1] + 1,
                     out_channels]
        # 檢查dilation和stride
    if dilation[0] != 1 and stride[0] != 1 or dilation[1] != 1 and stride[1] != 1:
        raise Exception('Dilation and stride of Conv2dTranspose cannot be unequal to 1 at the same time.')
    if shape_only:
        return out_shape
    # 分框架实现
    if framework == 'TensorFlow':
        params_str = '(filters={0}, kernel_size={1}, strides={2}, padding=\"{3}\", output_padding={4}, ' \
                     'dilation_rate={5}, use_bias={6})'.format(out_channels, kernel_size, stride,
                                                               padding, out_padding, dilation, bias)
        return eval('tf.keras.layers.Conv2DTranspose' + params_str), out_shape
    elif framework == 'PyTorch':
        if padding == 'valid':
            params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding=0, ' \
                         'output_padding={4}, dilation={5}, bias={6})'.format(in_channels, out_channels, kernel_size,
                                                                           stride, out_padding, dilation, bias)
            return eval('torch.nn.ConvTranspose2d' + params_str), out_shape
        else:
            # pre_input_shape = [input_shape[0] * 2, input_shape[1] * 2, out_channels]
            pre_input_shape = [input_shape[0] * stride[0], input_shape[1] * stride[1], out_channels]
            pad_h, pad_w = cal_same_padding(pre_input_shape, kernel_size, stride, dilation)
            params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding=({4}, {5}), ' \
                         'output_padding={6}, dilation={7}, bias={8})'.format(in_channels, out_channels, kernel_size,
                                                                           stride, pad_h // 2, pad_w // 2, out_padding,
                                                                           dilation, bias)
            return eval('torch.nn.ConvTranspose2d' + params_str), out_shape
    elif framework == 'Jittor':
        if padding == 'valid':
            params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding=0, ' \
                         'output_padding={4}, dilation={5}, bias={6})'.format(in_channels, out_channels, kernel_size,
                                                                           stride, out_padding, dilation, bias)
            return eval('jittor.nn.ConvTranspose' + params_str), out_shape
        else:
            # pre_input_shape = [input_shape[0] * 2, input_shape[1] * 2, out_channels]
            pre_input_shape = [input_shape[0] * stride[0], input_shape[1] * stride[1], out_channels]
            pad_h, pad_w = cal_same_padding(pre_input_shape, kernel_size, stride, dilation)
            params_str = '(in_channels={0}, out_channels={1}, kernel_size={2}, stride={3}, padding=({4}, {5}), ' \
                         'output_padding={6}, dilation={7}, bias={8})'.format(in_channels, out_channels, kernel_size,
                                                                           stride, pad_h // 2, pad_w // 2, out_padding,
                                                                           dilation, bias)
            return eval('jittor.nn.ConvTranspose' + params_str), out_shape
    else:
        raise Exception('No support DL framework.')

#
# import numpy as np
# params = {
#     'in_channels': 1,
#     'out_channels': 32,
#     'kernel_size': [3, 2],
#     'stride': [1, 1],
#     'padding': 'valid',
#     'dilation': [2, 3]
# }
# a, _ = get_op_and_shape([28, 28, 1], params, 'TensorFlow')
# print(a(np.ones((100, 28, 28, 1), dtype=np.float32)))
# tf.keras.layers.Conv2DTranspose()
