"""
    gather层
"""
import torch
import numpy as np


class Gather(torch.nn.Module):
    def __init__(self, dim, index, tensor_space):
        super(Gather, self).__init__()
        self.dim = dim
        self.index = torch.from_numpy(np.asarray(index)).long()
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
        self.tensor_space = tensor_space

    def forward(self, x):
        if x.ndim == self.tensor_space:
            x = eval(self.encoder[self.tensor_space])

        # 首先扩展index维度
        index = self.index
        for i in range(x.ndim):
            if i == self.dim or i == x.ndim + self.dim:
                continue
            index = torch.unsqueeze(index, i)

        sentence = 'index.expand('
        for i in range(x.ndim):
            if i == self.dim or i == x.ndim + self.dim:
                sentence = sentence + '-1, '
            else:
                sentence = sentence + str(x.size(i)) + ', '
        index_exp = eval(sentence[:-2] + ')')
        index_exp = index_exp.cuda()
        x = torch.gather(x, self.dim, index_exp)
        if x.ndim == self.tensor_space:
            x = eval(self.decoder[self.tensor_space])
        return x
