import tensorflow as tf
import torch
import jittor


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if shape_only:
        return input_shape
    # 字段读取或默认初始化
    # num_features
    num_features = params['num_features']
    # affine
    if params.__contains__('affine'):
        affine = params['affine']
    else:
        affine = True
    # track_running_stats
    if params.__contains__('track_running_stats'):
        track_running_stats = params['track_running_stats']
    else:
        track_running_stats = True
    if framework == 'TensorFlow':
        # eps
        if params.__contains__('eps'):
            eps = params['eps']
        else:
            eps = 1e-5
        # momentum
        if params.__contains__('momentum'):
            momentum = 1 - params['momentum']
        else:
            momentum = 0.9
        return tf.keras.layers.BatchNormalization(trainable=affine, epsilon=eps, momentum=momentum), input_shape
    elif framework == 'PyTorch':
        # eps
        if params.__contains__('eps'):
            eps = params['eps']
        else:
            eps = 1e-5
        # momentum
        if params.__contains__('momentum'):
            momentum = params['momentum']
        else:
            momentum = 0.1
        return torch.nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats), input_shape
    elif framework == 'Jittor':
        # eps
        if params.__contains__('eps'):
            eps = params['eps']
        else:
            eps = 1e-5
        # momentum
        if params.__contains__('momentum'):
            momentum = params['momentum']
        else:
            momentum = 0.1
        return jittor.nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats), input_shape
    else:
        raise Exception('No support DL framework.')

