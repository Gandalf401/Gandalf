"""
    用于池化、卷积等的same padding
    采用tensorflow的same padding定义
    output_shape = ceil(input_shape / stride)
"""


def cal_same_padding(input_shape, kernel_size, stride, dilation=(1, 1)):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    height = input_shape[0]
    width = input_shape[1]
    effective_kernel_size_h = (kernel_size[0] - 1) * dilation[0] + 1
    effective_kernel_size_w = (kernel_size[1] - 1) * dilation[1] + 1
    out_h = (height + stride[0] - 1) // stride[0]
    out_w = (width + stride[1] - 1) // stride[1]
    padding_h = max(0, (out_h - 1) * stride[0] + effective_kernel_size_h - height)
    padding_w = max(0, (out_w - 1) * stride[1] + effective_kernel_size_w - width)
    # if height % stride[0] == 0:
    #     padding_height = max(kernel_size[0] - stride[0], 0)
    # else:
    #     padding_height = max(kernel_size[0] - (height % stride[0]), 0)
    # if width % stride[1] == 0:
    #     padding_width = max(kernel_size[1] - stride[1], 0)
    # else:
    #     padding_width = max(kernel_size[1] - (width % stride[1]), 0)

    return padding_h, padding_w


def cal_same_padding_1d(input_shape, kernel_size, stride, dilation=1):
    size = input_shape[0]
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    out = (size + stride - 1) // stride
    padding = max(0, (out - 1) * stride + effective_kernel_size - size)

    return padding


def cal_same_padding_3d(input_shape, kernel_size, stride, dilation=(1, 1, 1)):
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    depth = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]
    effective_kernel_size_d = (kernel_size[0] - 1) * dilation[0] + 1
    effective_kernel_size_h = (kernel_size[1] - 1) * dilation[1] + 1
    effective_kernel_size_w = (kernel_size[2] - 1) * dilation[2] + 1
    out_d = (depth + stride[0] - 1) // stride[0]
    out_h = (height + stride[1] - 1) // stride[1]
    out_w = (width + stride[2] - 1) // stride[2]
    padding_d = max(0, (out_d - 1) * stride[0] + effective_kernel_size_d - depth)
    padding_h = max(0, (out_h - 1) * stride[1] + effective_kernel_size_h - height)
    padding_w = max(0, (out_w - 1) * stride[2] + effective_kernel_size_w - width)
    # if height % stride[0] == 0:
    #     padding_height = max(kernel_size[0] - stride[0], 0)
    # else:
    #     padding_height = max(kernel_size[0] - (height % stride[0]), 0)
    # if width % stride[1] == 0:
    #     padding_width = max(kernel_size[1] - stride[1], 0)
    # else:
    #     padding_width = max(kernel_size[1] - (width % stride[1]), 0)

    return padding_d, padding_h, padding_w
