import tensorflow as tf
import torch
import jittor

from src.configuration.support.tf_expansion.crop_and_resize import CropAndResize as tf_cr
from src.configuration.support.torch_expansion.crop_and_resize import CropAndResize as torch_cr
from src.configuration.support.jittor_expansion.crop_and_resize import CropAndResize as jittor_cr


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    output_shape = params['output_shape']
    if isinstance(output_shape, int):
        output_shape = (output_shape, output_shape)
    elif isinstance(output_shape, list):
        output_shape = (output_shape[0], output_shape[1])
    if params.__contains__('mode'):
        mode = params['mode']
    else:
        mode = 'nearest'
    top = params['top']
    left = params['left']
    height = params['height']
    width = params['width']

    output_shape_lst = [output_shape[0], output_shape[1], input_shape[-1]]
    if shape_only:
        return output_shape_lst

    if framework == 'TensorFlow':
        return tf_cr(output_shape, top, left, height, width, mode), output_shape_lst
    elif framework == 'PyTorch':
        return torch_cr(output_shape, top, left, height, width, mode), output_shape_lst
    elif framework == 'Jittor':
        return jittor_cr(output_shape, top, left, height, width, mode), output_shape_lst
    else:
        raise Exception('No support DL framework.')
