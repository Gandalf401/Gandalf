import tensorflow as tf
import torch
import jittor


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return input_shape
    if params.__contains__('dim'):
        dim = params['dim']
    else:
        dim = -1
    if framework == 'TensorFlow':
        return tf.keras.layers.Softmax(axis=dim), input_shape
    elif framework == 'PyTorch':
        return torch.nn.Softmax(dim=dim), input_shape
    elif framework == 'Jittor':
        return jittor.nn.Softmax(dim=dim), input_shape
    else:
        raise Exception('No support DL framework.')