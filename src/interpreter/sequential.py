'''
    框架无关的sequential模型
'''
import random

import tensorflow as tf
import torch
import jittor


def get_random_id():
    res = 0
    for i in range(4):
        res = res * 10 + random.randint(0, 9)
    return res


class SequentialAcrossFramework:
    def __init__(self, framework):
        self.framework = framework
        if framework == 'TensorFlow':
            self.object = tf.keras.Sequential()
        elif framework == 'PyTorch':
            self.object = torch.nn.Sequential()
        elif framework == 'Jittor':
            self.object = jittor.nn.Sequential()
        else:
            raise Exception('no support DL framework')

    def add_across_framework(self, layer):
        if self.framework == 'TensorFlow':
            self.object.add(layer)
        elif self.framework == 'PyTorch':
            self.object.add_module('layer_{0}'.format(get_random_id()), layer)
        elif self.framework == 'Jittor':
            self.object.add_module('layer_{0}'.format(get_random_id()), layer)
        else:
            raise Exception('no support DL framework')

