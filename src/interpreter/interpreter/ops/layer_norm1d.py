import tensorflow as tf
import torch
import jittor


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return input_shape
    # affine
    if params.__contains__('affine'):
        affine = params['affine']
    else:
        affine = True
    # eps
    if params.__contains__('eps'):
        eps = params['eps']
    else:
        eps = 1e-5
    if framework == 'TensorFlow':
        return tf.keras.layers.LayerNormalization(trainable=affine, epsilon=eps, axis=[1, 2]), input_shape
    elif framework == 'PyTorch':
        normalized_shape = [input_shape[-1], input_shape[1]]
        return torch.nn.LayerNorm(eps=eps, elementwise_affine=affine,
                                  normalized_shape=normalized_shape), input_shape
    elif framework == 'Jittor':
        normalized_shape = [input_shape[-1], input_shape[1]]
        return jittor.nn.LayerNorm(eps=eps, elementwise_affine=affine,
                                   normalized_shape=normalized_shape), input_shape
    else:
        raise Exception('No support DL framework.')
