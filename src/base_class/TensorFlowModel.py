"""
    继承自TensorFlow Model类的基类
    用于编程式模型定义
"""
import tensorflow as tf
from src.interpreter.interpreter.tf_interpreter import TFInterpreter


class TensorFlowModel(tf.keras.Model):
    interpreter = TFInterpreter()

    built_in = ['_TF_MODULE_IGNORED_PROPERTIES', '__call__', '__class__', '__delattr__', '__dict__', '__dir__',
                '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__',
                '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__',
                '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__',
                '_activity_regularizer', '_add_inbound_node', '_add_unique_metric_name',
                '_add_variable_with_custom_getter', '_assert_compile_was_called', '_assert_weights_created',
                '_autocast', '_base_init', '_build_model_with_inputs', '_cache_output_metric_attributes',
                '_call_accepts_kwargs', '_call_arg_was_passed', '_call_fn_args', '_callable_losses', '_check_call_args',
                '_check_trainable_weights_consistency', '_checkpoint_dependencies', '_clear_losses',
                '_collect_input_masks', '_compile_distribution', '_compile_eagerly', '_compile_from_inputs',
                '_compile_time_distribution_strategy', '_compile_weights_loss_and_weighted_metrics', '_compute_dtype',
                '_compute_output_and_mask_jointly', '_dedup_weights', '_deferred_dependencies',
                '_distribution_standardize_user_data', '_distribution_strategy', '_dtype', '_dtype_defaulted_to_floatx',
                '_dtype_policy', '_dynamic', '_eager_add_metric', '_eager_losses', '_expects_mask_arg',
                '_expects_training_arg', '_experimental_run_tf_function', '_feed_loss_fns', '_feed_output_names',
                '_feed_output_shapes', '_feed_sample_weights', '_feed_targets', '_flatten',
                '_gather_children_attribute', '_gather_saveables_for_checkpoint', '_get_call_arg_value',
                '_get_callback_model', '_get_existing_metric', '_get_node_attribute_at_index', '_get_trainable_state',
                '_get_training_eval_metrics', '_graph', '_graph_network_add_loss', '_graph_network_add_metric',
                '_handle_activity_regularization', '_handle_deferred_dependencies', '_handle_metrics',
                '_handle_per_output_metrics', '_handle_weight_regularization', '_inbound_nodes', '_init_call_fn_args',
                '_init_distributed_function_cache_if_not_compiled', '_init_graph_network', '_init_metric_attributes',
                '_init_set_name', '_init_subclassed_network', '_insert_layers', '_is_compiled', '_is_graph_network',
                '_is_layer', '_keras_api_names', '_keras_api_names_v1', '_layers',
                '_list_extra_dependencies_for_serialization', '_list_functions_for_serialization', '_lookup_dependency',
                '_loss_weights_list', '_losses', '_make_callback_model', '_make_execution_function',
                '_make_predict_function', '_make_test_function', '_make_train_function', '_maybe_build',
                '_maybe_cast_inputs', '_maybe_create_attribute', '_maybe_initialize_trackable',
                '_maybe_load_initial_epoch_from_ckpt', '_metrics', '_name', '_name_based_attribute_restore',
                '_name_based_restores', '_name_scope', '_no_dependency', '_non_trainable_weights',
                '_obj_reference_counts', '_obj_reference_counts_dict', '_object_identifier', '_outbound_nodes',
                '_output_loss_metrics', '_preload_simple_restoration', '_prepare_output_masks',
                '_prepare_sample_weights', '_prepare_skip_target_masks', '_prepare_total_loss',
                '_prepare_validation_data', '_process_target_tensor_for_compile',
                '_recompile_weights_loss_and_weighted_metrics', '_restore_from_checkpoint_position', '_reuse',
                '_run_eagerly', '_run_internal_graph', '_sample_weight_modes', '_scope', '_select_training_loop',
                '_self_name_based_restores', '_self_setattr_tracking', '_self_unconditional_checkpoint_dependencies',
                '_self_unconditional_deferred_dependencies', '_self_unconditional_dependency_names', '_self_update_uid',
                '_set_connectivity_metadata_', '_set_dtype_policy', '_set_input_attrs', '_set_inputs',
                '_set_mask_metadata', '_set_metric_attributes', '_set_optimizer', '_set_output_attrs',
                '_set_output_names', '_set_per_output_metric_attributes', '_set_trainable_state', '_setattr_tracking',
                '_should_compute_mask', '_single_restoration_from_checkpoint_position', '_standardize_user_data',
                '_symbolic_add_metric', '_symbolic_call', '_targets', '_tf_api_names', '_tf_api_names_v1',
                '_thread_local', '_track_layers', '_track_trackable', '_trackable_saved_model_saver',
                '_trackable_saver', '_tracking_metadata', '_trainable', '_trainable_weights',
                '_unconditional_checkpoint_dependencies', '_unconditional_dependency_names', '_undeduplicated_weights',
                '_update_sample_weight_modes', '_update_uid', '_updated_config', '_updates',
                '_validate_compile_param_for_distribution_strategy', '_validate_graph_inputs_and_outputs',
                '_validate_or_infer_batch_size', '_warn_about_input_casting', 'activity_regularizer', 'add_loss',
                'add_metric', 'add_update', 'add_variable', 'add_weight', 'apply', 'build', 'built', 'built_in', 'call',
                'compile', 'compute_mask', 'compute_output_shape', 'compute_output_signature', 'conv1', 'conv2',
                'count_params', 'dtype', 'dynamic', 'evaluate', 'evaluate_generator', 'fit', 'fit_generator', 'forward',
                'from_config', 'get_config', 'get_input_at', 'get_input_mask_at', 'get_input_shape_at', 'get_layer',
                'get_losses_for', 'get_output_at', 'get_output_mask_at', 'get_output_shape_at', 'get_updates_for',
                'get_weights', 'inbound_nodes', 'input', 'input_mask', 'input_shape', 'input_spec', 'inputs',
                'interpreter', 'layers', 'load_weights', 'losses', 'metrics', 'metrics_names', 'name', 'name_scope',
                'non_trainable_variables', 'non_trainable_weights', 'optimizer', 'outbound_nodes', 'output',
                'output_mask', 'output_shape', 'outputs', 'predict', 'predict_generator', 'predict_on_batch', 'produce',
                'reset_metrics', 'reset_states', 'run_eagerly', 'sample_weights', 'save', 'save_weights', 'set_weights',
                'state_updates', 'stateful', 'submodules', 'summary', 'supports_masking', 'test_on_batch', 'to_json',
                'to_yaml', 'train_on_batch', 'trainable', 'trainable_variables', 'trainable_weights', 'updates',
                'variables', 'weights', 'with_name_scope', 'fill']

    def __init__(self):
        super(TensorFlowModel, self).__init__()

    def forward(self, x, **kwargs):
        return x

    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def call(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def produce(self):
        # 提取用户后来自定义的内容
        interpreter = TensorFlowModel.interpreter
        custom_items = set(dir(self)) - set(TensorFlowModel.built_in)
        print(custom_items)
        # 过滤方法
        for item in custom_items:
            if isinstance(getattr(self, item), dict) and len(getattr(self, item)) != 0:
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