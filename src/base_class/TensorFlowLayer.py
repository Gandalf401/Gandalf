"""
    继承自TensorFlow Layer类的基类
    用于编程式模型定义
"""
import tensorflow as tf
from src.interpreter.interpreter.tf_interpreter import TFInterpreter


class TensorFlowLayer(tf.keras.layers.Layer):
    interpreter = TFInterpreter()

    built_in = ['_TF_MODULE_IGNORED_PROPERTIES', '__call__', '__class__', '__delattr__', '__dict__', '__dir__',
                 '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__',
                 '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__',
                 '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__',
                 '__weakref__', '_activity_regularizer', '_add_inbound_node', '_add_variable_with_custom_getter',
                 '_autocast', '_call_accepts_kwargs', '_call_arg_was_passed', '_call_fn_args', '_callable_losses',
                 '_checkpoint_dependencies', '_clear_losses', '_collect_input_masks', '_compute_dtype',
                 '_dedup_weights', '_deferred_dependencies', '_dtype', '_dtype_defaulted_to_floatx', '_dtype_policy',
                 '_dynamic', '_eager_add_metric', '_eager_losses', '_expects_mask_arg', '_expects_training_arg',
                 '_flatten', '_gather_children_attribute', '_gather_saveables_for_checkpoint', '_get_call_arg_value',
                 '_get_existing_metric', '_get_node_attribute_at_index', '_get_trainable_state',
                 '_handle_activity_regularization', '_handle_deferred_dependencies', '_handle_weight_regularization',
                 '_inbound_nodes', '_init_call_fn_args', '_init_set_name', '_initial_weights', '_input_spec',
                 '_is_layer', '_keras_api_names', '_keras_api_names_v1', '_layers',
                 '_list_extra_dependencies_for_serialization', '_list_functions_for_serialization',
                 '_lookup_dependency', '_losses', '_maybe_build', '_maybe_cast_inputs', '_maybe_create_attribute',
                 '_maybe_initialize_trackable', '_metrics', '_name', '_name_based_attribute_restore',
                 '_name_based_restores', '_name_scope', '_no_dependency', '_non_trainable_weights',
                 '_obj_reference_counts', '_object_identifier', '_outbound_nodes', '_preload_simple_restoration',
                 '_restore_from_checkpoint_position', '_self_setattr_tracking', '_set_connectivity_metadata_',
                 '_set_dtype_policy', '_set_mask_metadata', '_set_trainable_state', '_setattr_tracking',
                 '_should_compute_mask', '_single_restoration_from_checkpoint_position', '_symbolic_add_metric',
                 '_symbolic_call', '_tf_api_names', '_tf_api_names_v1', '_thread_local', '_track_trackable',
                 '_trackable_saved_model_saver', '_tracking_metadata', '_trainable', '_trainable_weights',
                 '_unconditional_checkpoint_dependencies', '_unconditional_dependency_names', '_update_uid',
                 '_updates', '_warn_about_input_casting', 'activity_regularizer', 'add_loss', 'add_metric',
                 'add_update', 'add_variable', 'add_weight', 'apply', 'build', 'built', 'call', 'compute_mask',
                 'compute_output_shape', 'compute_output_signature', 'count_params', 'dtype', 'dynamic', 'forward',
                 'from_config', 'get_config', 'get_input_at', 'get_input_mask_at', 'get_input_shape_at',
                 'get_losses_for', 'get_output_at', 'get_output_mask_at', 'get_output_shape_at', 'get_updates_for',
                 'get_weights', 'inbound_nodes', 'input', 'input_mask', 'input_shape', 'input_spec', 'interpreter',
                 'losses', 'metrics', 'name', 'name_scope', 'non_trainable_variables', 'non_trainable_weights',
                 'outbound_nodes', 'output', 'output_mask', 'output_shape', 'produce', 'set_weights', 'stateful',
                 'submodules', 'supports_masking', 'trainable', 'trainable_variables', 'trainable_weights', 'updates',
                 'variables', 'weights', 'with_name_scope', '_obj_reference_counts_dict', 'built_in',
                '_self_unconditional_deferred_dependencies', '_self_unconditional_dependency_names',
                '_self_name_based_restores', '_self_update_uid', '_self_unconditional_checkpoint_dependencies', 'fill']

    def __init__(self):
        super(TensorFlowLayer, self).__init__()

    def build(self, input_shape):
        super(TensorFlowLayer, self).build(input_shape)

    def forward(self, x, **kwargs):
        return x

    def call(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def produce(self):
        # 提取用户后来自定义的内容
        interpreter = TensorFlowLayer.interpreter
        custom_items = set(dir(self)) - set(TensorFlowLayer.built_in)
        # 过滤方法
        for item in custom_items:
            if isinstance(getattr(self, item), dict) and len(getattr(self, item)) != 0:
                # 属性
                layer = getattr(self, item)
                print(layer)
                print(item)
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
            "float": tf.float32,
            "float32": tf.float32,
            "float64": tf.float64,
            "int": tf.int32,
            "int8": tf.int8,
            "int16": tf.int16,
            "int32": tf.int32,
            "int64": tf.int64
        }
        return tf.fill(shape, value, dtype=dtype_table[dtype])