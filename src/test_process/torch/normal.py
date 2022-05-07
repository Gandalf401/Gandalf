import logging

import torch
from torch.utils.data import DataLoader

from src.metircs.metric_tables import get_torch_metrics


def normal_test(model, ds_test, metrics=['OneHotAccuracy']):
    # 生成metrics存储字典
    metrics_dict_test = {}
    for m in metrics:
        metrics_dict_test[m] = get_torch_metrics(m)
    logging.info('[JsonDL] Preparing for your dataset, it may take some time...')
    # 准备数据集
    test_sample_num = len(ds_test)
    ds_test_loader = DataLoader(ds_test, 1, False)
    logging.info('[JsonDL] Dataset has been ready.')
    model.eval()
    # 开始测试
    i = 0
    for x, y in ds_test_loader:
        print('Sample={0}/{1}, testing...'.format(i + 1, test_sample_num))
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        x = torch.autograd.Variable(x)
        y = torch.autograd.Variable(y)
        predicts = model(x)
        for m in metrics_dict_test.values():
            m.update_state(y, predicts)
        i += 1

    # 输出测试的
    info = '{0} Samples has been tested'.format(test_sample_num)
    for m in metrics:
        info = info + ', {0}:{1}'.format(m, metrics_dict_test[m].result())
    print(info)
    # 输出结束
    tmp_test = {}
    for m in metrics:
        tmp_test[m] = float(metrics_dict_test[m].result().numpy())

    return tmp_test


