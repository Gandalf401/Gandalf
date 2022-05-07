"""
    top_k层
"""
import torch


class TopK(torch.nn.Module):
    def __init__(self, k, output='indices'):
        super(TopK, self).__init__()
        self.k = k
        self.output = output
        if output == 'indices':
            self.sentence = 'x.topk(self.k).indices'
        elif output == 'values':
            self.sentence = 'x.topk(self.k).values'
        else:
            self.sentence = 'x.topk(self.k)'

    def forward(self, x):
        return eval(self.sentence)
