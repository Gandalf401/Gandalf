import tensorflow as tf
import torch
import jittor

from src.configuration.support.torch_expansion.gaussian_noise import GaussianNoise as torch_gaussian_noise
from src.configuration.support.jittor_expansion.gaussian_noise import GaussianNoise as jittor_gaussian_noise


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return input_shape

    stddev = params['stddev']

    if framework == 'TensorFlow':
        return tf.keras.layers.GaussianNoise(stddev), input_shape
    elif framework == 'PyTorch':
        return torch_gaussian_noise(stddev), input_shape
    elif framework == 'Jittor':
        return jittor_gaussian_noise(stddev), input_shape
    else:
        raise Exception('No support DL framework.')