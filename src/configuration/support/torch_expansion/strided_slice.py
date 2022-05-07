"""
    strided_slice层
"""
import torch


class StridedSlice(torch.nn.Module):
    def __init__(self, begin, end, stride, tensor_space):
        super(StridedSlice, self).__init__()
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
        # sentences
        sentences = []
        left = 0
        while left <= len(begin) - 1:
            s = 'x[' + ':, ' * left + 'range({0}, {1}, {2}), '.format(begin[left], end[left],
                                                                      stride[left]) + ':, ' * (len(begin) - 1 - left)
            s = s[:-2] + ']'
            sentences.append(s)
            left += 1
        self.sentences = sentences

    def forward(self, x):
        if x.ndim == self.tensor_space:
            x = eval(self.encoder[self.tensor_space])
        for s in self.sentences:
            x = eval(s)
        if x.ndim == self.tensor_space:
            x = eval(self.decoder[self.tensor_space])
        return x
