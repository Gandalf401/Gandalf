import torch.nn as nn
from itertools import repeat


class SpatialDropout2d(nn.Module):
    """
    空间dropout，即在指定轴方向上进行dropout，常用于Embedding层和CNN层后
    如对于(batch, timesteps, embedding)的输入，若沿着axis=1则可对embedding的若干channel进行整体dropout
    若沿着axis=2则可对某些token进行整体dropout
    """

    def __init__(self, drop=0.5):
        super(SpatialDropout2d, self).__init__()
        self.drop = drop

    def forward(self, x):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple, 应当与inputs的shape一致，其中值为1的即沿着drop的轴
        """
        outputs = x.clone()
        noise_shape = (x.shape[0], *repeat(1, x.dim() - 2), x.shape[-1])  # 默认沿着中间所有的shape

        if not self.training or self.drop == 0:
            return x
        else:
            noises = self._make_noises(x, noise_shape)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(x)
            outputs.mul_(noises)
            return outputs

    def _make_noises(self, x, noise_shape):
        return x.new().resize_(noise_shape)
