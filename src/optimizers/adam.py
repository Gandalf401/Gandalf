import tensorflow as tf
import torch
import jittor


def get_optimizer(framework, params, network_parameters=None):
    if params.__contains__('lr'):
        lr = params['lr']
    else:
        lr = 0.001
    if params.__contains__('beta1'):
        beta1 = params['beta1']
    else:
        beta1 = 0.9
    if params.__contains__('beta2'):
        beta2 = params['beta2']
    else:
        beta2 = 0.999
    if params.__contains__('eps'):
        eps = params['eps']
    else:
        eps = 1e-8
    if params.__contains__('decay'):
        decay = params['decay']
    else:
        decay = 0.0
    if framework == 'TensorFlow':
        return tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2, epsilon=eps, decay=decay)
    elif framework == 'PyTorch':
        return torch.optim.Adam(network_parameters, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=decay)
    elif framework == 'Jittor':
        return jittor.optim.Adam(network_parameters, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=decay)
    else:
        raise Exception('No support DL framework.')
