"""
    自实现的topk-accuracy metrics
"""
import numpy as np
import torch


class TopKAccuracy:
    def __init__(self, k):
        self.true_example_num = 0
        self.total_example_num = 0
        self.k = k

    def update_state(self, y, predicts):
        topk = predicts.topk(self.k).indices
        predict_res = (y == topk).float().max(1)
        self.true_example_num += predict_res.sum()
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
