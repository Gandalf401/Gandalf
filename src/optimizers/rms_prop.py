import tensorflow as tf
import torch
import jittor


def get_optimizer(framework, params, network_parameters=None):
    if params.__contains__('lr'):
        lr = params['lr']
    else:
        lr = 0.01
    if params.__contains__('momentum'):
        momentum = params['momentum']
    else:
        momentum = 0
    if params.__contains__('centered'):
        centered = params['centered']
    else:
        centered = False
    if params.__contains__('eps'):
        eps = params['eps']
    else:
        eps = 1e-8
    if framework == 'TensorFlow':
        return tf.keras.optimizers.RMSprop(learning_rate=lr, centered=centered, momentum=momentum, epsilon=eps)
    elif framework == 'PyTorch':
        return torch.optim.RMSprop(network_parameters, lr=lr, centered=centered, momentum=momentum, eps=eps)
    elif framework == 'Jittor':
        return jittor.optim.RMSprop(network_parameters, lr=lr, centered=centered, momentum=momentum, eps=eps)
    else:
        raise Exception('No support DL framework.')
