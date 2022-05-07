"""
    计算broadcast shape
"""


def cal_broadcast_shape(t1, t2):
    output_shape = []
    for i in range(0, min(len(t1), len(t2))):
        if t1[len(t1) - i - 1] != t2[len(t2) - i - 1] and min(t1[len(t1) - i - 1], t2[len(t2) - i - 1]) != 1:
            raise Exception('Some tensor shapes cannot be broadcast.')
        output_shape.append(max(t1[len(t1) - i - 1], t2[len(t2) - i - 1]))
    if len(t1) <= len(t2):
        for i in range(len(t1) + 1, len(t2) + 1):
            output_shape.append(t2[len(t2) - i])
    else:
        for i in range(len(t2) + 1, len(t1) + 1):
            output_shape.append(t1[len(t1) - i])
    output_shape.reverse()
    return output_shape

