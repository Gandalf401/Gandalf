"""
    tile层
"""
import torch


class Tile(torch.nn.Module):
    def __init__(self, multiples, tensor_space):
        super(Tile, self).__init__()
        self.multiples = multiples
        self.tensor_space = tensor_space
        self.sentence = 'x.repeat('
        for x in multiples:
            self.sentence = self.sentence + str(x) + ', '
        self.sentence = self.sentence[:-2] + ')'
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
        x = eval(self.sentence)
        if x.ndim == self.tensor_space:
            x = eval(self.decoder[self.tensor_space])
        return x
