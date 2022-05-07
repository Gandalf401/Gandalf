"""
    a typical Conv2D module implemented by PyTorch
"""
import torch


class Conv2dModule(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        self.conv3x3 = torch.nn.Conv2d(input_channels, output_channels,
                                       (3, 3), (1, 1), (0, 0), (1, 1))
        self.batch_norm2d = torch.nn.BatchNorm2d(output_channels, 1e-5, 0.1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.batch_norm2d(x)
        x = self.relu(x)
        return x
