import tensorflow as tf
import torch
import jittor


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    pad = params['pad']
    if isinstance(pad, int):
        pad = [pad, pad, pad, pad]
    else:
        if len(pad) == 2:
            pad = [pad[0], pad[0], pad[1], pad[1]]
        elif len(pad) != 4:
            raise Exception('Param \"pad\" of ZeroPad2d has the wrong length.')
    output_shape = [pad[0] + pad[1] + input_shape[0], pad[2] + pad[3] + input_shape[1], input_shape[-1]]
    if shape_only:
        return output_shape
    if framework == 'TensorFlow':
        return tf.keras.layers.ZeroPadding2D(padding=((pad[2], pad[3]), (pad[0], pad[1]))), output_shape
    elif framework == 'PyTorch':
        return torch.nn.ZeroPad2d(padding=pad), output_shape
    elif framework == 'Jittor':
        return jittor.nn.ZeroPad2d(padding=pad), output_shape
    else:
        raise Exception('No support DL framework.')
