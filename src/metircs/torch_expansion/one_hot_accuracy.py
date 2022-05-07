"""
    自实现的one-hot accuracy metrics
"""
import numpy as np
import torch


class OneHotAccuracy:
    def __init__(self):
        self.true_example_num = 0
        self.total_example_num = 0

    def update_state(self, y, predicts):
        category = predicts.argmax(1)
        category_y = y.argmax(1)
        self.true_example_num += (category_y == category).sum()
        self.total_example_num += y.size(0)

    def reset_states(self):
        self.true_example_num = 0
        self.total_example_num = 0

    def result(self):
        if self.total_example_num == 0:
            x = np.asarray(0.)
            return torch.from_numpy(x).to(torch.float32)
        else:
            return self.true_example_num / self.total_example_num
