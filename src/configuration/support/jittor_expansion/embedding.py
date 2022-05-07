"""
    embedding层
"""
import jittor


class Embedding(jittor.nn.Module):
    def __init__(self, input_dim, output_dim, mask_zero):
        super(Embedding, self).__init__()
        self.input_dim = input_dim
        self.mask_zero = mask_zero
        self.linear = jittor.nn.Linear(input_dim, output_dim, bias=False)

    def execute(self, x):
        # 制作one_hot编码 => (batch, seq, input_dim)
        x = jittor.unsqueeze(x, -1)
        x = jittor.zeros((x.shape[0], x.shape[1], self.input_dim)).scatter(-1, x, jittor.ones(x.shape))
        # 如果要屏蔽zero标签作为padding 需要mask
        if self.mask_zero:
            x[:, :, 0] = 0.0
        # 通过全链接层 => (batch, seq, output_dim)
        return self.linear(x)
