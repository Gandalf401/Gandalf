import tensorflow as tf
import torch
import jittor

from src.configuration.support.torch_expansion.global_max_pool1d import GlobalMaxPool1d as torch_maxpool
from src.configuration.support.jittor_expansion.global_max_pool1d import GlobalMaxPool1d as jt_maxpool


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return [input_shape[-1]]
    if framework == 'TensorFlow':
        return tf.keras.layers.GlobalMaxPooling1D(), [input_shape[-1]]
    elif framework == 'PyTorch':
        return torch_maxpool(), [input_shape[-1]]
    elif framework == 'Jittor':
        return jt_maxpool(), [input_shape[-1]]
    else:
        raise Exception('No support DL framework.')