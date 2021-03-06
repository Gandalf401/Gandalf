"""
    网络层的规范配置
"""
from types import FunctionType


layers_required = {
    "Conv1d": {"in_channels": int, "out_channels": int, "kernel_size": int},
    "Conv2d": {"in_channels": int, "out_channels": int, "kernel_size": (int, list)},
    "Conv3d": {"in_channels": int, "out_channels": int, "kernel_size": (int, list)},
    "Conv2dTranspose": {"in_channels": int, "out_channels": int, "kernel_size": (int, list)},
    "Conv3dTranspose": {"in_channels": int, "out_channels": int, "kernel_size": (int, list)},
    "DepthwiseConv1d": {"in_channels": int, "kernel_size": int},
    "SeparableConv1d": {"in_channels": int, "out_channels": int, "kernel_size": int},
    "DepthwiseConv2d": {"in_channels": int, "kernel_size": (int, list)},
    "SeparableConv2d": {"in_channels": int, "out_channels": int, "kernel_size": (int, list)},
    "BatchNorm1d": {"num_features": int},
    "BatchNorm2d": {"num_features": int},
    "BatchNorm3d": {"num_features": int},
    "LayerNorm1d": {},
    "LayerNorm2d": {},
    "LayerNorm3d": {},

    "Embedding": {"input_dim": int, "output_dim": int},

    "MaxPool1d": {"pool_size": int},
    "AvgPool1d": {"pool_size": int},
    "MaxPool2d": {"pool_size": (int, list)},
    "AvgPool2d": {"pool_size": (int, list)},
    "MaxPool3d": {"pool_size": (int, list)},
    "AvgPool3d": {"pool_size": (int, list)},
    "GlobalMaxPool1d": {},
    "GlobalMaxPool2d": {},
    "GlobalMaxPool3d": {},
    "GlobalAvgPool1d": {},
    "GlobalAvgPool2d": {},
    "GlobalAvgPool3d": {},

    "ZeroPadding2d": {"pad": (int, list)},

    "Flatten": {},
    "Reshape": {"output_shape": list},
    "Unsqueeze": {"dim": int},
    "Squeeze": {"dim": int},
    "Transpose": {"output_shape": list},

    "Resize": {"output_shape": (int, list)},
    "Upsample1d": {"scale_factor": (int, list)},
    "Upsample2d": {"scale_factor": (int, list)},
    "Upsample3d": {"scale_factor": (int, list)},
    "CropAndResize": {"output_shape": (int, list), "top": int, "left": int, "height": int, "width": int},

    "Dense": {"in_features": int, "out_features": int},
    "BiasAdd": {"bias": (int, float)},

    "Shape": {},
    "Gather": {"index": list},

    "Dropout": {},

    "ReduceMean": {},
    "ReduceMax": {},
    "ReduceSum": {},
    "ReduceProd": {},

    "Argmax": {},
    "Argmin": {},

    "Cast": {},

    "Ceil": {},
    "Floor": {},
    "Exp": {},
    "Compare": {},
    "Sqrt": {},
    "Square": {},
    "Rsqrt": {},
    "Tile": {"multiples": list},
    "TopK": {"k": int},
    "Slice": {"begin": list, "size": list},
    "StridedSlice": {"begin": list, "end": list},

    "Lambda": {"function": (str, FunctionType), "output_shape": list},
    "GaussianNoise": {"stddev": (int, float)},

    "Threshold": {"threshold": (int, float)},
    "ReLU": {},
    "ReLU6": {},
    "PReLU": {},
    "LeakyReLU": {},
    "ELU": {},
    "SeLU": {},
    "Sigmoid": {},
    "Tanh": {},
    "Softmax": {},

    "Add": {},
    "Subtract": {},
    "Multiply": {},
    "Divide": {},
    "Maximum": {},
    "Minimum": {},
    "Average": {},
    "Concatenate": {},

    "Repeat": {"n": int},

    "RNN": {"input_size": int, "hidden_size": int},
    "GRU": {"input_size": int, "hidden_size": int},
    "LSTM": {"input_size": int, "hidden_size": int},
    "RNNCell": {"input_size": int, "hidden_size": int},
    "GRUCell": {"input_size": int, "hidden_size": int},
    "LSTMCell": {"input_size": int, "hidden_size": int},
    "BiRNN": {"input_size": int, "hidden_size": int},
    "BiGRU": {"input_size": int, "hidden_size": int},
    "BiLSTM": {"input_size": int, "hidden_size": int},
}

layers_optional = {
    "Conv1d": {"stride": int, "dilation": int, "bias": bool},
    "Conv2d": {"stride": (int, list), "bias": bool, "dilation": (int, list)},
    "Conv3d": {"stride": (int, list), "bias": bool, "dilation": (int, list)},
    "Conv2dTranspose": {"stride": (int, list), "out_padding": (int, list), "dilation": (int, list), "bias": bool},
    "Conv3dTranspose": {"stride": (int, list), "out_padding": (int, list), "dilation": (int, list), "bias": bool},
    "DepthwiseConv1d": {"stride": int, "dilation": int, "depth_multiplier": int, "bias": bool},
    "SeparableConv1d": {"stride": int, "dilation": int, "depth_multiplier": int, "bias": bool},
    "DepthwiseConv2d": {"stride": (int, list), "dilation": (int, list), "depth_multiplier": int, "bias": bool},
    "SeparableConv2d": {"stride": (int, list), "dilation": (int, list), "depth_multiplier": int, "bias": bool},
    "BatchNorm1d": {"eps": float, "momentum": float, "affine": bool, "track_running_stats": bool},
    "BatchNorm2d": {"eps": float, "momentum": float, "affine": bool, "track_running_stats": bool},
    "BatchNorm3d": {"eps": float, "momentum": float, "affine": bool, "track_running_stats": bool},
    "LayerNorm1d": {"eps": float, "affine": bool},
    "LayerNorm2d": {"eps": float, "affine": bool},
    "LayerNorm3d": {"eps": float, "affine": bool},

    "Embedding": {"mask_zero": bool},

    "MaxPool1d": {"stride": int},
    "AvgPool1d": {"stride": int},
    "MaxPool2d": {"stride": (int, list)},
    "AvgPool2d": {"stride": (int, list)},
    "MaxPool3d": {"stride": (int, list)},
    "AvgPool3d": {"stride": (int, list)},
    "GlobalMaxPool1d": {},
    "GlobalMaxPool2d": {},
    "GlobalMaxPool3d": {},
    "GlobalAvgPool1d": {},
    "GlobalAvgPool2d": {},
    "GlobalAvgPool3d": {},

    "ZeroPadding2d": {},

    "Flatten": {},
    "Reshape":{"channel_stay": bool},
    "Unsqueeze":{"channel_stay": bool},
    "Squeeze":{"channel_stay": bool},
    "Transpose":{"channel_stay": bool},

    "Resize": {},
    "Upsample1d": {},
    "Upsample2d": {},
    "Upsample3d": {},
    "CropAndResize": {},

    "Dense": {"bias": bool},
    "BiasAdd": {},

    "Shape": {},
    "Gather": {"dim": int},

    "ReduceMean": {"dim": int, "keep_dims": bool},
    "ReduceMax": {"dim": int, "keep_dims": bool},
    "ReduceSum": {"dim": int, "keep_dims": bool},
    "ReduceProd": {"dim": int, "keep_dims": bool},

    "Argmax": {"dim": int},
    "Argmin": {"dim": int},

    "Dropout": {"p": (float, int)},

    "Cast": {},

    "Repeat": {},

    "Ceil": {},
    "Floor": {},
    "Exp": {},
    "Compare": {},
    "Sqrt": {},
    "Square": {},
    "Rsqrt": {},
    "Tile": {"channel_stay": bool},
    "TopK": {},
    "Slice": {"channel_stay": bool},
    "StridedSlice": {"channel_stay": bool, "stride": list},

    "Lambda": {},
    "GaussianNoise": {},

    "Threshold": {},
    "ReLU": {},
    "ReLU6": {},
    "LeakyReLU": {"alpha": (float, int)},
    "PReLU": {"alpha": (float, int), "share": bool},
    "ELU": {"alpha": (float, int)},
    "SeLU": {},
    "Sigmoid": {},
    "Tanh": {},
    "Softmax": {"dim": int},

    "Add": {},
    "Subtract": {},
    "Multiply": {},
    "Divide": {},
    "Maximum": {},
    "Minimum": {},
    "Average": {},
    "Concatenate": {"dim": int},

    "RNN": {"bias": bool, "dropout": (float, int)},
    "GRU": {"bias": bool, "dropout": (float, int)},
    "LSTM": {"bias": bool, "dropout": (float, int)},
    "RNNCell": {"bias": bool},
    "GRUCell": {"bias": bool},
    "LSTMCell": {"bias": bool},
    "BiRNN": {"bias": bool, "dropout": (float, int)},
    "BiGRU": {"bias": bool, "dropout": (float, int)},
    "BiLSTM": {"bias": bool, "dropout": (float, int)},
}

layers_range = {
    "Conv1d": {"padding": ['same', 'valid', 'causal']},
    "Conv2d": {"padding": ['same', 'valid']},
    "Conv3d": {"padding": ['same', 'valid']},
    "Conv2dTranspose": {"padding": ['same', 'valid']},
    "Conv3dTranspose": {"padding": ['same', 'valid']},
    "DepthwiseConv1d": {"padding": ['same', 'valid']},
    "SeparableConv1d": {"padding": ['same', 'valid']},
    "DepthwiseConv2d": {"padding": ['same', 'valid']},
    "SeparableConv2d": {"padding": ['same', 'valid']},
    "BatchNorm1d": {},
    "BatchNorm2d": {},
    "BatchNorm3d": {},
    "LayerNorm1d": {},
    "LayerNorm2d": {},
    "LayerNorm3d": {},

    "Embedding": {},

    "MaxPool1d": {"padding": ['same', 'valid']},
    "AvgPool1d": {"padding": ['same', 'valid']},
    "MaxPool2d": {"padding": ['same', 'valid']},
    "AvgPool2d": {"padding": ['same', 'valid']},
    "MaxPool3d": {"padding": ['same', 'valid']},
    "AvgPool3d": {"padding": ['same', 'valid']},
    "GlobalMaxPool1d": {},
    "GlobalMaxPool2d": {},
    "GlobalMaxPool3d": {},
    "GlobalAvgPool1d": {},
    "GlobalAvgPool2d": {},
    "GlobalAvgPool3d": {},

    "ZeroPadding2d": {},

    "Flatten": {},
    "Reshape": {},
    "Unsqueeze": {},
    "Squeeze": {},
    "Transpose": {},

    "Resize": {"mode": ['nearest', 'bilinear', 'bicubic']},
    "Upsample1d": {"mode": ['nearest', 'bilinear']},
    "Upsample2d": {"mode": ['nearest', 'bilinear']},
    "Upsample3d": {"mode": ['nearest', 'bilinear']},
    "CropAndResize": {"mode": ['nearest', 'bilinear']},

    "Dense": {},
    "BiasAdd": {},

    "Shape": {},
    "Gather": {},

    "ReduceMean": {},
    "ReduceMax": {},
    "ReduceSum": {},
    "ReduceProd": {},

    "Argmax": {},
    "Argmin": {},

    "Dropout": {},

    "Cast": {"target_dtype": ["float32", "float64", "int8", "int16", "int32", "int64", "float", "int"]},

    "Ceil": {},
    "Floor": {},
    "Exp": {},
    "Compare": {"op": ["<", "<=", "==", ">=", ">"]},
    "Sqrt": {},
    "Square": {},
    "Rsqrt": {},
    "Tile": {},
    "TopK": {"output": ["indices", "values", "all"]},
    "Slice": {},
    "StridedSlice": {},

    "Lambda": {},
    "GaussianNoise": {},

    "Threshold": {},
    "ReLU": {},
    "ReLU6": {},
    "PReLU": {},
    "LeakyReLU": {},
    "ELU": {},
    "SeLU": {},
    "Sigmoid": {},
    "Tanh": {},
    "Softmax": {},

    "Add": {},
    "Subtract": {},
    "Multiply": {},
    "Divide": {},
    "Maximum": {},
    "Minimum": {},
    "Average": {},
    "Concatenate": {},

    "Repeat": {},

    "RNN": {},
    "GRU": {},
    "LSTM": {},
    "RNNCell": {},
    "GRUCell": {},
    "LSTMCell": {},
    "BiRNN": {},
    "BiGRU": {},
    "BiLSTM": {},
}
