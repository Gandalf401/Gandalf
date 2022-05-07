"""
    jittor自定义flatten层
"""
import jittor


class Flatten(jittor.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def execute(self, x, start_dim=1, end_dim=-1):
        return jittor.flatten(x, start_dim, end_dim)
