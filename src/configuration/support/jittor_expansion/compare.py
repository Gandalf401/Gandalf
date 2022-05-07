"""
    compare层 各类张量元素级比较的封装层
"""
import jittor


class Compare(jittor.nn.Module):
    def __init__(self, op):
        super(Compare, self).__init__()
        op_table = {
            ">": "jittor.greater(x, y)",
            ">=": "jittor.greater_equal(x, y)",
            "==": "jittor.equal(x, y)",
            "<=": "jittor.less_equal(x, y)",
            "<": "jittor.less(x, y)",
        }
        self.op = op_table[op]

    def execute(self, x, y):
        return eval(self.op)