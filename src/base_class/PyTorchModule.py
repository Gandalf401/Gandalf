"""
    继承自PyTorch Module类的基类
    用于编程式模型定义
"""
import torch
from src.interpreter.interpreter.torch_interpreter import TorchInterpreter


class PyTorchModule(torch.nn.Module):
    interpreter = TorchInterpreter()

    built_in = ['T_destination', '__annotations__', '__call__', '__class__', '__delattr__', '__dict__', '__dir__',
                 '__doc__', '__eq__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__gt__', '__hash__',
                 '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__',
                 '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__',
                 '__subclasshook__', '__weakref__', '_apply', '_call_impl', '_get_backward_hooks', '_get_name',
                 '_load_from_state_dict', '_maybe_warn_non_full_backward_hook', '_named_members',
                 '_register_load_state_dict_pre_hook', '_register_state_dict_hook', '_replicate_for_data_parallel',
                 '_save_to_state_dict', '_slow_forward', '_version', 'add_module', 'apply', 'bfloat16', 'buffers',
                 'built_in', 'children', 'cpu', 'cuda', 'double', 'dump_patches', 'eval', 'extra_repr', 'float',
                 'forward', 'half', 'interpreter', 'load_state_dict', 'modules', 'named_buffers', 'named_children',
                 'named_modules', 'named_parameters', 'parameters', 'produce', 'register_backward_hook',
                 'register_buffer', 'register_forward_hook', 'register_forward_pre_hook',
                 'register_full_backward_hook', 'register_parameter', 'requires_grad_', 'share_memory', 'state_dict',
                 'to', 'train', 'type', 'xpu', 'zero_grad', 'fill']

    def __init__(self):
        super(PyTorchModule, self).__init__()

    def forward(self, x, **kwargs):
        return x

    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def produce(self):
        # 提取用户后来自定义的内容
        interpreter = PyTorchModule.interpreter
        custom_items = set(dir(self)) - set(PyTorchModule.built_in)
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
            "float": torch.float32,
            "float32": torch.float32,
            "float64": torch.float64,
            "int": torch.int32,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64
        }
        return torch.full(shape, value, dtype=dtype_table[dtype])