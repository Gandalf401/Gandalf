"""
    继承自Jittor Module类的基类
    用于编程式模型定义
"""
import jittor
from src.interpreter.interpreter.jittor_interpreter import JittorInterpreter


class JittorModule(jittor.nn.Module):
    interpreter = JittorInterpreter()

    built_in = ['__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__',
                '__ge__', '__getattribute__', '__gt__', '__hash__', '__hooked_call__', '__init__', '__init_subclass__',
                '__le__', '__lt__', '__module__', '__name__', '__ne__', '__new__', '__reduce__', '__reduce_ex__',
                '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_get_name',
                '_place_hooker', 'apply', 'children', 'dfs', 'eval', 'execute', 'extra_repr', 'forward', 'interpreter',
                'is_training', 'load', 'load_parameters', 'load_state_dict', 'modules', 'mpi_param_broadcast',
                'named_modules', 'named_parameters', 'parameters', 'produce', 'register_backward_hook',
                'register_forward_hook', 'register_input_backward_hook', 'register_output_backward_hook',
                'register_pre_forward_hook', 'remove_backward_hook', 'remove_forward_hook',
                'remove_input_backward_hook', 'remove_output_backward_hook', 'remove_pre_forward_hook',
                'requires_grad_', 'save', 'state_dict', 'train', 'built_in', 'fill']

    def __init__(self):
        super(JittorModule, self).__init__()

    def forward(self, x, **kwargs):
        return x

    def execute(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def produce(self):
        # 提取用户后来自定义的内容
        interpreter = JittorModule.interpreter
        custom_items = set(dir(self)) - set(JittorModule.built_in)
        # 过滤方法
        for item in custom_items:
            if isinstance(getattr(self, item), dict):
                # 属性
                layer = getattr(self, item)
                layer_name = layer['name']
                input_shape = layer['input_shape']
                params = layer['params'] if layer.__contains__('params') else {}
                op, _ = interpreter.get_op_and_shape(layer_name, params, input_shape)
                setattr(self, item, op)

    # 以下为内置方法
    # 是一些内置的深度学习操作函数

    # fill 填充
    def fill(self, shape, value, dtype="float32"):
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
        return jittor.full(shape, value, dtype=dtype_table[dtype])