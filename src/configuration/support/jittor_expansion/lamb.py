"""
    lambda层
"""
import jittor
from jittor.transform import Lambda as lamb


class Lambda(jittor.nn.Module):
    def __init__(self, function):
        super(Lambda, self).__init__()
        self.lamb = lamb(function)

    def execute(self, x):
        return self.lamb(x)
