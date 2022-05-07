"""
    torch下same padding的奇数pad情况扩展padding3d
"""
import torch
import torch.nn.functional as F


class SamePadding3d(torch.nn.Module):
    def __init__(self, depth_pad, height_pad, width_pad):
        super(SamePadding3d, self).__init__()
        self.pad = [0, int(width_pad % 2 != 0),
                    0, int(height_pad % 2 != 0),
                    0, int(depth_pad % 2 != 0)]

    def forward(self, x):
        return F.pad(x, self.pad)
