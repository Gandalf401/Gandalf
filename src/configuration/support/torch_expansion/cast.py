"""
    cast层
"""
import torch


class Cast(torch.nn.Module):
    def __init__(self, t_dtype):
        super(Cast, self).__init__()
        dtype_table = {
            "float": torch.float32,
            "float32": torch.float32,
            "float64": torch.float64,
            "int": torch.int32,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64
        }
        self.t_dtype = dtype_table[t_dtype]

    def forward(self, x):
        return x.to(self.t_dtype)
