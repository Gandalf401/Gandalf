"""
    lambda层
"""
import torch
from torchvision.transforms import Lambda as lamb


class Lambda(torch.nn.Module):
    def __init__(self, function):
        super(Lambda, self).__init__()
        self.lamb = lamb(function)

    def forward(self, x):
        return self.lamb(x)
