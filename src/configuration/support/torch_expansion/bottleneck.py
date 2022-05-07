"""
    bottleneck模块，用于branch_to和index
"""
import math

import torch
import torch.nn.functional as F


class Bottleneck(torch.nn.Module):
    def __init__(self, shape_needed, shape_r):
        super(Bottleneck, self).__init__()
        if shape_needed[0] >= shape_r[0]:
            h_diff = shape_needed[0] - shape_r[0]
            w_diff = shape_needed[1] - shape_r[1]
            self.pad = [w_diff // 2, math.ceil(w_diff / 2), h_diff // 2, math.ceil(h_diff / 2)]
            if shape_needed[-1] == shape_r[-1]:
                self.conv11 = None
            else:
                self.conv11 = torch.nn.Conv2d(shape_r[-1], shape_needed[-1], 1, 1)
        else:
            s = shape_r[0] // shape_needed[0] + 1
            h_diff = shape_needed[0] * s - shape_r[0]
            w_diff = shape_needed[1] * s - shape_r[1]
            self.pad = [w_diff // 2, math.ceil(w_diff / 2), h_diff // 2, math.ceil(h_diff / 2)]
            self.conv11 = torch.nn.Conv2d(shape_r[-1], shape_needed[-1], 1, s)

    def forward(self, x):
        x, residual = x
        residual = F.pad(residual, self.pad)
        if self.conv11 is not None:
            residual = self.conv11(residual)
        x = torch.add(x, residual)
        return x
