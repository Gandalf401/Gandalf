"""
    torch下same padding的奇数pad情况扩展padding1d
"""
import torch


class SamePadding1d(torch.nn.Module):
    def __init__(self, pad):
        super(SamePadding1d, self).__init__()
        self.pad = [0, int(pad % 2 != 0)]

    def forward(self, x):
        return torch.nn.pad(x, self.pad)
