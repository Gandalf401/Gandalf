import tensorflow as tf
import torch
import jittor


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    if params.__contains__('bias'):
        bias = params['bias']
    else:
        bias = True
    in_features = params['in_features']
    out_features = params['out_features']

    output_shape = list(input_shape)
    output_shape[-1] = out_features
    if shape_only:
        return output_shape
    if framework == 'TensorFlow':
        return tf.keras.layers.Dense(units=out_features, use_bias=bias), output_shape
    elif framework == 'PyTorch':
        return torch.nn.Linear(in_features, out_features, bias), output_shape
    elif framework == 'Jittor':
        return jittor.nn.Linear(in_features, out_features, bias), output_shape
    else:
        raise Exception('No support DL framework.')