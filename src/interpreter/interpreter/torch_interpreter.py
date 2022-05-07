"""
    torch框架翻译器
"""
import src.interpreter.interpreter.ops as ops
from src.configuration.ops.methods_table import methods_table


class TorchInterpreter:
    def get_op_and_shape(self, name, params, current_shape):
        return eval(methods_table[name] + '(current_shape, params, \"PyTorch\")')
