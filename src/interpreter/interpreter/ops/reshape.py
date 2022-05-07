import tensorflow as tf
import torch
import jittor

from src.configuration.support.torch_expansion.reshape import Reshape as torch_reshape
from src.configuration.support.jittor_expansion.reshape import Reshape as jt_reshape


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    output_shape = params['output_shape']
    tensor_space = params['tensor_space']
    output_shape_lst = list(output_shape)
    idx = -1
    for i in range(len(output_shape)):
        if output_shape[i] == -1:
            if idx != -1:
                raise Exception('Only one -1 can be involved in your target shape.')
            idx = i
    if idx != -1:
        prod = 1
        for x in input_shape:
            prod *= x
        for i in range(len(output_shape)):
            if i != idx:
                prod = prod // output_shape[i]
        output_shape_lst[idx] = prod
    if shape_only:
        return output_shape_lst
    if framework == 'TensorFlow':
        return tf.keras.layers.Reshape(output_shape), output_shape_lst
    elif framework == 'PyTorch':
        target_shape = tuple([-1] + output_shape_lst)
        return torch_reshape(target_shape, tensor_space), output_shape_lst
    elif framework == 'Jittor':
        target_shape = tuple([-1] + output_shape_lst)
        return jt_reshape(target_shape, tensor_space), output_shape_lst
    else:
        raise Exception('No support DL framework.')
