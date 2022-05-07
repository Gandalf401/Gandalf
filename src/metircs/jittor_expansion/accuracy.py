"""
    自实现的accuracy metrics
"""
import jittor
import numpy as np


class Accuracy:
    def __init__(self):
        self.true_example_num = 0
        self.total_example_num = 0

    def update_state(self, y, predicts):
        category = predicts.argmax(1)[0]
        self.true_example_num += (y == category).sum().numpy()
        self.total_example_num += y.size(0)

    def reset_states(self):
        self.true_example_num = 0
        self.total_example_num = 0

    def result(self):
        if self.total_example_num == 0:
            return np.zeros(1, dtype=np.float32)
        else:
            return self.true_example_num / self.total_example_num
