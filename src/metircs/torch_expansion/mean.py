"""
    自实现的mean metrics
"""
import numpy as np
import torch


class Mean:
    def __init__(self):
        self.data_lst = []

    def update_state(self, x):
        self.data_lst.append(x)

    def reset_states(self):
        self.data_lst = []

    def result(self):
        if len(self.data_lst) == 0:
            x = np.asarray(0.)
            return torch.from_numpy(x).to(torch.float32)
        else:
            res = 0
            for x in self.data_lst:
                res += x
            return res / len(self.data_lst)
