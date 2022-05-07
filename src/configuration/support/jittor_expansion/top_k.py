"""
    top_k层
"""
import jittor


class TopK(jittor.nn.Module):
    def __init__(self, k, output='indices'):
        super(TopK, self).__init__()
        self.k = k
        self.output = output
        if output == 'indices':
            self.sentence = 'x.topk(self.k)[1]'
        elif output == 'values':
            self.sentence = 'x.topk(self.k)[0]'
        else:
            self.sentence = 'x.topk(self.k)'

    def execute(self, x):
        return eval(self.sentence)
