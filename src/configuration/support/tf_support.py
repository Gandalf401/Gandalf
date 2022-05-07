'''
    当前支持的tensorflow算子及其转换规则
    （已废弃）
'''


class TFSupport:
    comparison_table = {
        # 批标准化
        'BatchNorm2d': 'tf.keras.layers.BatchNormalization',
        # 卷积层
        'Conv1d': 'tf.keras.layers.Conv1D',
        'Conv2d': 'tf.keras.layers.Conv2D',
        'Conv3d': 'tf.keras.layers.Conv3D',
        'ConvTranspose2d': 'tf.keras.layers.Conv2DTranspose',
        # 转换层
        'Flatten': 'tf.keras.layers.Flatten',
        # 全链接层
        'Dense': 'tf.keras.layers.Dense',
        # 补丁层
        'ZeroPadding1d': 'tf.keras.layers.ZeroPadding1D',
        'ZeroPadding2d': 'tf.keras.layers.ZeroPadding2D',
        'ZeroPadding3d': 'tf.keras.layers.ZeroPadding3D',
        # 池化层
        'MaxPool1d': 'tf.keras.layers.MaxPooling1D',
        'AvgPool1d': 'tf.keras.layers.AvgPooling1D',
        'MaxPool2d': 'tf.keras.layers.MaxPooling2D',
        'AvgPool2d': 'tf.keras.layers.AvgPooling2D',
        'MaxPool3d': 'tf.keras.layers.MaxPooling3D',
        'AvgPool3d': 'tf.keras.layers.AvgPooling3D',
        'GlobalMaxPool2d': 'tf.keras.layers.GlobalMaxPooling2D',
        'GlobalAvgPool2d': 'tf.keras.layers.GlobalAvgPooling2D',
        # 循环层
        'RNN': 'tf.keras.layers.RNN',
        # 'SimpleRNN': 'tf.keras.layers.SimpleRNN',
        'GRU': 'tf.keras.layers.GRU',
        'LSTM': 'tf.keras.layers.LSTM',
        # 'ConvLSTM2d': 'tf.keras.layers.ConvLSTM2D',
        # 嵌入层
        'Embedding': 'torch.nn.Embedding',
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
        'ReLU': 'tf.keras.layers.ReLU',
        'ELU': 'tf.keras.layers.ELU',
        'ReLU6': 'tf.keras.layers.ReLU(6.)',
        'PReLU': 'tf.keras.layers.PReLU',
        'LeakyReLU': 'tf.keras.layers.LeakyReLU',
        'Softmax': 'tf.keras.layers.Softmax',
        'Sigmoid': 'tf.keras.layers.Activation("sigmoid")',
        'Tanh': 'tf.keras.layers.Activation("tanh")',
        'Threshold': 'tf.keras.layers.ThresholdReLU',
        # # 封装层
        # 'TimeDistributed': 'tf.keras.layers.TimeDistributed',
        # 'Bidirectional': 'tf.keras.layers.Bidirectional'
    }
