"""
    selu层
"""
import jittor


class SeLU(jittor.nn.Module):
    def __init__(self):
        super(SeLU, self).__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.lamb = 1.0507009873554804934193349852946

    def execute(self, x):
        return self.lamb * jittor.nn.elu(x, self.alpha)
