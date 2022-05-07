"""
    slice层
"""
import jittor


class Slice(jittor.nn.Module):
    def __init__(self, begin, size, tensor_space):
        super(Slice, self).__init__()
        self.sentence = 'x['
        for i in range(len(begin)):
            self.sentence = self.sentence + '{0}:{1}, '.format(begin[i], begin[i] + size[i])
        self.sentence = self.sentence[:-2] + ']'
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

    def execute(self, x):
        if x.ndim == self.tensor_space:
            x = eval(self.encoder[self.tensor_space])
        x = eval(self.sentence)
        if x.ndim == self.tensor_space:
            x = eval(self.decoder[self.tensor_space])
        return x
