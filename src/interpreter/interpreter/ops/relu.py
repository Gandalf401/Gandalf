import tensorflow as tf
import torch
import jittor


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return input_shape
    if framework == 'TensorFlow':
        return tf.keras.layers.ReLU(), input_shape
    elif framework == 'PyTorch':
        return torch.nn.ReLU(), input_shape
    elif framework == 'Jittor':
        return jittor.nn.ReLU(), input_shape
    else:
        raise Exception('No support DL framework.')
