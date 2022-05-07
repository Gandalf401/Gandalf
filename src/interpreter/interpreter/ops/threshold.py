import tensorflow as tf
import torch
import jittor

from src.configuration.support.jittor_expansion.threshold import Threshold as jt_threshold


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return input_shape
    threshold = params['threshold']

    if framework == 'TensorFlow':
        return tf.keras.layers.ThresholdedReLU(threshold), input_shape
    elif framework == 'PyTorch':
        return torch.nn.Threshold(threshold, 0.), input_shape
    elif framework == 'Jittor':
        return jt_threshold(threshold), input_shape
    else:
        raise Exception('No support DL framework.')
