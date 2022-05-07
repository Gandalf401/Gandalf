"""
    tf框架翻译器
"""
import src.interpreter.interpreter.ops as ops
from src.configuration.ops.methods_table import methods_table


class TFInterpreter:
    def get_op_and_shape(self, name, params, current_shape):
        return eval(methods_table[name] + '(current_shape, params, \"TensorFlow\")')
        # if len(params) == 0:
        #     return eval(TFSupport.comparison_table[name] + '()')
        # else:
        #     tmp = ''
        #     for k, v in params.items():
        #         if isinstance(v, str):
        #             tmp = tmp + ', {0}="{1}"'.format(k, v)
        #         else:
        #             tmp = tmp + ', {0}={1}'.format(k, v)
        #     param_str = tmp
        #     if name in TFInterpreter.special_lst:
        #         op_str = TFSupport.comparison_table[name]
        #         op_str = op_str[:-1] + param_str + ')'
        #     else:
        #         op_str = TFSupport.comparison_table[name]
        #         op_str = op_str + '(' + param_str[2:] + ')'
        #     return eval(op_str)
