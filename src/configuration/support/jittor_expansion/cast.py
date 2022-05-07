"""
    cast层
"""

import jittor


class Cast(jittor.nn.Module):
    def __init__(self, t_dtype):
        super(Cast, self).__init__()
        dtype_table = {
            "float": jittor.float32,
            "float32": jittor.float32,
            "float64": jittor.float64,
            "int": jittor.int32,
            "int8": jittor.int8,
            "int16": jittor.int16,
            "int32": jittor.int32,
            "int64": jittor.int64
        }
        self.t_dtype = dtype_table[t_dtype]

    def execute(self, x):
        return jittor.cast(x, self.t_dtype)
