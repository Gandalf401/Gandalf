"""
    crop_and_resize层
"""
import jittor
from jittor.transform import Crop


class CropAndResize(jittor.nn.Module):
    def __init__(self, size, top, left, height, width, mode):
        super(CropAndResize, self).__init__()
        self.resize_layer = jittor.nn.Resize(size, mode, align_corners=True)
        self.crop_layer = Crop(top, left, height, width)

    def execute(self, x):
        x = self.crop_layer(x)
        return self.resize_layer(x)
