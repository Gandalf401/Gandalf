"""
    自实现的mae metrics
"""
import numpy as np
import torch


class MAE:
    def __init__(self):
        self.error_lst = []

    def update_state(self, y, predicts):
        self.error_lst.append((y - predicts))

    def reset_states(self):
        self.error_lst = []

    def result(self):
        if len(self.error_lst) == 0:
            x = np.asarray(0.)
            return torch.from_numpy(x).to(torch.float32)
        else:
            res = 0
            for err in self.error_lst:
                err_abs = err.abs()
                res += err_abs.mean()
            return res / len(self.error_lst)
