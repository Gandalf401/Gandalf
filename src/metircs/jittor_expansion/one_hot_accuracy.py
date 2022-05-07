"""
    自实现的one-hot accuracy metrics
"""
import numpy as np
import jittor


class OneHotAccuracy:
    def __init__(self):
        self.true_example_num = 0
        self.total_example_num = 0

    def update_state(self, y, predicts):
        category = predicts.argmax(1)[0]
        category_y = y.argmax(1)[0]
        self.true_example_num += (category_y == category).sum().data
        self.total_example_num += y.size(0)

    def reset_states(self):
        self.true_example_num = 0
        self.total_example_num = 0

    def result(self):
        if self.total_example_num == 0:
            return np.zeros(1, dtype=np.float32)
        else:
            return self.true_example_num / self.total_example_num


y = jittor.Var([[1., 0., 0.]])
predict = jittor.Var([[0., 0.5, 0.5]])
a = OneHotAccuracy()
a.update_state(y, predict)

