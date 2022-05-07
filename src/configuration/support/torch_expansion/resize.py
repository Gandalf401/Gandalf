"""
    resize层
"""
import torch
from torchvision.transforms import Resize as torch_resize
from torchvision.transforms.functional import InterpolationMode


class Resize(torch.nn.Module):
    def __init__(self, size, mode):
        super(Resize, self).__init__()
        mode_table = {
            'bilinear': InterpolationMode.BILINEAR,
            'nearest': InterpolationMode.NEAREST,
            'bicubic': InterpolationMode.BICUBIC
        }
        self.resize_layer = torch_resize(size, mode_table[mode])

    def forward(self, x):
        return self.resize_layer(x)
