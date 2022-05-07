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
    if params.__contains__('nesterov'):
        nesterov = params['nesterov']
    else:
        nesterov = False
    if params.__contains__('decay'):
        decay = params['decay']
    else:
        decay = 0.0
    if framework == 'TensorFlow':
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=nesterov, decay=decay)
    elif framework == 'PyTorch':
        return torch.optim.SGD(network_parameters, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=decay)
    elif framework == 'Jittor':
        return jittor.optim.SGD(network_parameters, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=decay)
    else:
        raise Exception('No support DL framework.')

