"""
    repeat层
"""
import jittor


class Repeat(jittor.nn.Module):
    def __init__(self, n):
        super(Repeat, self).__init__()
        self.n = n

    def execute(self, x):
        batch_size = x.size(0)
        x = x.repeat(1, self.n)
        x = jittor.unsqueeze(x, 1)
        x = jittor.reshape(x, (batch_size, self.n, -1))
        return x
