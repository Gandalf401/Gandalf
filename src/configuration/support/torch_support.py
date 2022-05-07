'''
    当前支持的pytorch算子及其转换规则
    （已废弃）
'''


class TorchSupport:
    comparison_table = {
        # 批标准化
        'BatchNorm2d': 'torch.nn.BatchNorm2d',
        # 卷积层
        'Conv1d': 'torch.nn.Conv1d',
        'Conv2d': 'torch.nn.Conv2d',
        'Conv3d': 'torch.nn.Conv3d',
        'ConvTranspose2d': 'torch.nn.ConvTranspose2d',
        # 转换层
        'Flatten': 'torch_expansion.flatten.Flatten',
        # 全链接层
        'Dense': 'torch.nn.Linear',
        # 补丁层
        'ZeroPadding1d': 'torch.nn.ZeroPad1d',
        'ZeroPadding2d': 'torch.nn.ZeroPad2d',
        'ZeroPadding3d': 'torch.nn.ZeroPad3d',
        # 池化层
        'MaxPool1d': 'torch.nn.MaxPool1d',
        'AvgPool1d': 'torch.nn.AvgPool1d',
        'MaxPool2d': 'torch.nn.MaxPool2d',
        'AvgPool2d': 'torch.nn.AvgPool2d',
        'MaxPool3d': 'torch.nn.MaxPool3d',
        'AvgPool3d': 'torch.nn.AvgPool3d',
        'GlobalMaxPool2d': 'torch_expansion.global_max_pool2d.GlobalMaxPool2d',
        'GlobalAvgPool2d': 'torch_expansion.global_avg_pool2d.GlobalAvgPool2d',
        # 循环层
        'RNN': 'torch.nn.RNN',
        'GRU': 'torch.nn.GRU',
        'LSTM': 'torch.nn.LSTM',
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
        'ReLU': 'torch.nn.ReLU',
        'ELU': 'torch.nn.ELU',
        'ReLU6': 'torch.nn.ReLU6',
        'PReLU': 'torch.nn.PReLU',
        'LeakyReLU': 'torch.nn.LeakyReLU',
        'Softmax': 'torch.nn.Softmax',
        'Sigmoid': 'torch.nn.Sigmoid',
        'Tanh': 'torch.nn.Tanh',
        'Threshold': 'torch.nn.Threshold',
    }
