import tensorflow as tf
import torch
import jittor

from src.configuration.support.jittor_expansion.embedding import Embedding


def get_op_and_shape(input_shape, params, framework, shape_only=False):
    input_dim = params['input_dim']
    output_dim = params['output_dim']
    if params.__contains__('mask_zero'):
        mask_zero = params['mask_zero']
    else:
        mask_zero = False

    output_shape = list(input_shape)
    output_shape.append(output_dim)

    if shape_only:
        return output_shape

    if framework == 'TensorFlow':
        return tf.keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim, mask_zero=mask_zero), output_shape
    elif framework == 'PyTorch':
        padding_idx = 0 if mask_zero else None
        return torch.nn.Embedding(num_embeddings=input_dim, embedding_dim=output_dim, padding_idx=padding_idx), \
               output_shape
    elif framework == 'Jittor':
        return Embedding(input_dim=input_dim, output_dim=output_dim, mask_zero=mask_zero), output_shape
    else:
        raise Exception('No support DL framework.')
