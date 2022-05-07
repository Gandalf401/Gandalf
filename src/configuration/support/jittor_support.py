"""
    当前支持的jittor算子及其转换规则
    （已废弃）
"""


class JittorSupport:
    comparison_table = {
        # 批标准化
        'BatchNorm2d': 'jittor.nn.BatchNorm2d',
        # 卷积层
        'Conv1d': 'jittor.nn.Conv1d',
        'Conv2d': 'jittor.nn.Conv2d',
        'Conv3d': 'jittor.nn.Conv3d',
        'ConvTranspose2d': 'jittor.nn.ConvTranspose2d',
        # 转换层
        'Flatten': 'torch_expansion.flatten.Flatten',
        # 全链接层
        'Dense': 'jittor.nn.Linear',
        # 补丁层
        'ZeroPadding1d': 'jittor.nn.ZeroPad1d',
        'ZeroPadding2d': 'jittor.nn.ZeroPad2d',
        'ZeroPadding3d': 'jittor.nn.ZeroPad3d',
        # 池化层
        'MaxPool1d': 'jittor.nn.MaxPool1d',
        'AvgPool1d': 'jittor.nn.AvgPool1d',
        'MaxPool2d': 'jittor.nn.MaxPool2d',
        'AvgPool2d': 'jittor.nn.AvgPool2d',
        'MaxPool3d': 'jittor.nn.MaxPool3d',
        'AvgPool3d': 'jittor.nn.AvgPool3d',
        'GlobalMaxPool2d': 'torch_expansion.global_max_pool2d.GlobalMaxPool2d',
        'GlobalAvgPool2d': 'torch_expansion.global_avg_pool2d.GlobalAvgPool2d',
        # 循环层
        'RNN': 'jittor.nn.RNN',
        'GRU': 'jittor.nn.GRU',
        'LSTM': 'jittor.nn.LSTM',
        # 嵌入层
        'Embedding': 'tf.keras.layers.Embedding',
        # 融合层
        'Add': 'tf.keras.layers.Add',
        'Subtract': 'tf.keras.layers.Subtract',
        'Multiply': 'tf.keras.layers.Multiply',
        'Maximum': 'tf.keras.layers.Maximum',
        'Minimum': 'tf.keras.layers.Minimum',
        'Dot': 'tf.keras.layers.Dot',
        'Average': 'tf.keras.layers.Average',
        'Concatenate': 'tf.keras.layers.Concatenate',
        # 激活层
        'ReLU': 'jittor.nn.ReLU',
        'ELU': 'jittor.nn.ELU',
        'ReLU6': 'jittor.nn.ReLU6',
        'PReLU': 'jittor.nn.PReLU',
        'LeakyReLU': 'jittor.nn.LeakyReLU',
        'Softmax': 'jittor.nn.Softmax',
        'Sigmoid': 'jittor.nn.Sigmoid',
        'Tanh': 'jittor.nn.Tanh',
        'Threshold': 'jittor.nn.Threshold',
    }
