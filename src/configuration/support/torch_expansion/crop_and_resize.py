"""
    crop_and_resize层
"""
import torch
import torchvision.transforms.functional
from torchvision.transforms import Resize as torch_resize
from torchvision.transforms.functional import InterpolationMode


class CropAndResize(torch.nn.Module):
    def __init__(self, size, top, left, height, width, mode):
        super(CropAndResize, self).__init__()
        mode_table = {
            'bilinear': InterpolationMode.BILINEAR,
            'nearest': InterpolationMode.NEAREST
        }
        self.resize_layer = torch_resize(size, mode_table[mode])
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, x):
        x = torchvision.transforms.functional.crop(x, self.top, self.left, self.width, self.height)
        return self.resize_layer(x)
