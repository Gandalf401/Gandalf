"""
    compare层 各类张量元素级比较的封装层
"""
import torch


class Compare(torch.nn.Module):
    def __init__(self, op):
        super(Compare, self).__init__()
        op_table = {
            ">": "torch.gt(x, y)",
            ">=": "torch.ge(x, y)",
            "==": "torch.equal(x, y)",
            "<=": "torch.le(x, y)",
            "<": "torch.less(x, y)",
        }
        self.op = op_table[op]

    def forward(self, x, y):
        return eval(self.op)