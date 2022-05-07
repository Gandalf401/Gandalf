"""
    transpose层
"""

import torch


class Transpose(torch.nn.Module):
    def __init__(self, output_shape, tensor_space):
        super(Transpose, self).__init__()
        self.output_shape = output_shape
        self.tensor_space = tensor_space
        self.encoder = {
            3: 'x.permute(0, 2, 1)',
            4: 'x.permute(0, 2, 3, 1)',
            5: 'x.permute(0, 2, 3, 4, 1)'
        }
        self.decoder = {
            3: 'x.permute(0, 2, 1)',
            4: 'x.permute(0, 3, 1, 2)',
            5: 'x.permute(0, 4, 1, 2, 3)'
        }
        sentence = 'x.permute('
        for x in self.output_shape:
            sentence = sentence + str(x) + ', '
        self.sentence = sentence[:-2] + ')'

    def forward(self, x):
        if x.ndim == self.tensor_space:
            x = eval(self.encoder[self.tensor_space])
        x = eval(self.sentence)
        if x.ndim == self.tensor_space:
            x = eval(self.decoder[self.tensor_space])
        return x


