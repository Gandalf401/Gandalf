import tensorflow as tf
import torch
import jittor
import torchvision.transforms.functional

from src.configuration.support.torch_expansion.add import Add as t_add
from src.configuration.support.jittor_expansion.add import Add as jt_add
from src.interpreter.interpreter.ops._cal_broadcast_shape import cal_broadcast_shape


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    t1 = input_shape[0]
    t2 = input_shape[1]
    if not isinstance(t1, list) or not isinstance(t2, list):
        raise Exception('Input shape of layer Add should be a list of lists.')
    # if t1[0] != t2[0]:
    #     raise Exception('Input tensors of layer Add should be with the same batch size.')
    output_shape = cal_broadcast_shape(t1, t2)
    if shape_only:
        return output_shape
    for i in range(2, len(input_shape)):
        if not isinstance(input_shape[i], list):
            raise Exception('Input shape of layer Add should be a list of lists.')
        output_shape = cal_broadcast_shape(output_shape, input_shape[i])
    if framework == 'TensorFlow':
        return tf.keras.layers.Add(), output_shape
    elif framework == 'PyTorch':
        return t_add(), output_shape
    elif framework == 'Jittor':
        return jt_add(), output_shape
    else:
        raise Exception('No support DL framework.')