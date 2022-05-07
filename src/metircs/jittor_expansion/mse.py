"""
    自实现的mse metrics
"""
import jittor
import numpy as np


class MSE:
    def __init__(self):
        self.error_lst = []

    def update_state(self, y, predicts):
        self.error_lst.append((y - predicts))

    def reset_states(self):
        self.error_lst = []

    def result(self):
        if len(self.error_lst) == 0:
            return np.zeros(1, dtype=np.float32)
        else:
            res = 0
            for err in self.error_lst:
                err_2 = err ** 2
                res += err_2.mean().numpy()
            return res / len(self.error_lst)
