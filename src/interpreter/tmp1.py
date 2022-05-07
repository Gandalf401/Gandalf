"""
    a typical Conv2D module
    implemented by PyTorch
"""
import torch


class Conv2dModule(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        self.conv3x3 = torch.nn.Conv2d(input_channels, output_channels,
                                       (3, 3), (1, 1), (0, 0), (1, 1))
        self.batch_norm2d = torch.nn.BatchNorm2d(output_channels,
                                                 1e-5, 0.1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.batch_norm2d(x)
        x = self.relu(x)
        return x

#
# import tensorflow as tf
#
#
# def get_conv2d_module(input_shape, output_channels):
#     inp = tf.keras.layers.Input(input_shape)
#     conv3x3 = tf.keras.layers.Conv2D(output_channels, 3, 1, 'valid')
#     batch_norm2d = tf.keras.layers.BatchNormalization(-1, 0.99, 1e-3)
#     relu = tf.keras.layers.ReLU()
#
#     fx = conv3x3(inp)
#     fx = batch_norm2d(fx)
#     fx = relu(fx)
#
#     model = tf.keras.Model(inp, fx)
#     return model
