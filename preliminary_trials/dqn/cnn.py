from src.dataset.get_dataset import get_dataset
from src.model_json import ModelJSON
from src.train_process.tf.normal import normal_train
from src.test_process.tf.normal import normal_test

import os
import tensorflow as tf
import numpy as np


def __generate_one_hot(y):
    materials = np.eye(10)
    return materials[y]


if __name__ == '__main__':
    model = ModelJSON('./mnist_cnn.json')

    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1))
    y = __generate_one_hot(y_train)
    y = y.astype(np.float32)
    x = x_train / 255.0
    x = x.astype(np.float32)
    data = tf.data.Dataset.from_tensor_slices((x, y))

    optimizer = {
        'name': 'Adam',
        'params': {'lr': 1e-4}
    }

    r1, r2 = normal_train(model.network, optimizer, "CrossEntropy", 5, 100, data, None, localized=False)
    print(r1)
    print(r2)
