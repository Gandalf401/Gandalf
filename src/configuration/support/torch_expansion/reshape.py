"""
 torch reshape层
"""
import torch


class Reshape(torch.nn.Module):
    def __init__(self, output_shape, tensor_space):
        super(Reshape, self).__init__()
        # e.g. (-1, 14, 14, 4)
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

    def forward(self, x):
        if x.ndim == self.tensor_space:
            x = eval(self.encoder[self.tensor_space])
        x = torch.reshape(x, self.output_shape)
        if x.ndim == self.tensor_space:
            x = eval(self.decoder[self.tensor_space])
        return x
