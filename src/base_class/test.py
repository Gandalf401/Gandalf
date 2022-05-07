from src.base_class import TensorFlowLayer, TensorFlowModel

import numpy as np


class CNNModule(TensorFlowLayer):
    def __init__(self):
        super(CNNModule, self).__init__()
        self.conv1 = {
            'name': 'Conv2d',
            'params': {'in_channels': 1, 'out_channels': 32, 'kernel_size': 3, 'padding': 'same'},
            'input_shape': [28, 28, 1]
        }
        self.avg_pool1 = {
            'name': 'AvgPool2d',
            'params': {'pool_size': 3, 'padding': 'same'},
            'input_shape': [28, 28, 1]
        }
        self.relu = {
            'name': 'ReLU',
            'input_shape': [28, 28, 1]
        }
        self.produce()

    def forward(self, x):
        x = self.conv1(x)
        x = self.avg_pool1(x)
        x = self.relu(x)
        return x


class CNN(TensorFlowModel):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = CNNModule()
        self.conv2 = CNNModule()
        self.add = {
            'name': 'Add'
        }
        self.produce()

    def forward(self, x):
        x1, x2 = x
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x = sel.add([x1, x2])
        return x


if __name__ == '__main__':
    cnn = CNN()
    x = np.ones((4, 28, 28, 1))
    print(cnn(x))