import tensorflow as tf
import torch
import jittor


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return input_shape
    if params.__contains__('p'):
        p = params['p']
    else:
        p = 0.5
    if framework == 'TensorFlow':
        return tf.keras.layers.Dropout(rate=p), input_shape
    elif framework == 'PyTorch':
        return torch.nn.Dropout(p=p), input_shape
    elif framework == 'Jittor':
        return jittor.nn.Dropout(p=p, is_train=True), input_shape
    else:
        raise Exception('No support DL framework.')