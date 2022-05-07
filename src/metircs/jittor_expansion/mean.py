"""
    自实现的mean metrics
"""
import jittor
import numpy as np


class Mean:
    def __init__(self):
        self.data_lst = []

    def update_state(self, x):
        self.data_lst.append(x)

    def reset_states(self):
        self.data_lst = []

    def result(self):
        if len(self.data_lst) == 0:
            return np.zeros(1, dtype=np.float32)
        else:
            res = 0
            for x in self.data_lst:
                res += x.numpy()
            return res / len(self.data_lst)
