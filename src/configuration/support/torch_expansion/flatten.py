"""
    torch自定义flatten层
"""
import torch


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x, start_dim=1, end_dim=-1):
        return torch.flatten(x, start_dim, end_dim)
