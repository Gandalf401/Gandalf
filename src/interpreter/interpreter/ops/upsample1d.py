import tensorflow as tf
import torch
import jittor


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    scale_factor = params['scale_factor']
    if params.__contains__('mode'):
        mode = params['mode']
    else:
        mode = 'nearest'

    output_shape = [input_shape[0] * scale_factor, input_shape[-1]]
    if shape_only:
        return output_shape
    if framework == 'TensorFlow':
        return tf.keras.layers.UpSampling1D(size=scale_factor, interpolation=mode), output_shape
    elif framework == 'PyTorch':
        return torch.nn.Upsample(scale_factor=scale_factor, mode=mode), output_shape
    elif framework == 'Jittor':
        return jittor.nn.Upsample(scale_factor=scale_factor, mode=mode), output_shape
    else:
        raise Exception('No support DL framework.')
