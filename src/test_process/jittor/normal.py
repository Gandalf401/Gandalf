import logging

import jittor

from src.metircs.metric_tables import get_jittor_metrics


def normal_test(model, ds_test, metrics=['OneHotAccuracy']):
    # 生成metrics存储字典
    metrics_dict_test = {}
    for m in metrics:
        metrics_dict_test[m] = get_jittor_metrics(m)
    logging.info('[JsonDL] Preparing for your dataset, it may take some time...')
    # 准备数据集
    ds_test = ds_test.set_attrs(batch_size=1, shuffle=False, drop_last=False)
    test_sample_num = len(ds_test)
    logging.info('[JsonDL] Dataset has been ready.')
    # 开始测试
    i = 0
    model.eval()
    for x, y in ds_test:
        print('Sample={0}/{1}, testing...'.format(i + 1, test_sample_num))
        out = model(x)
        for m in metrics_dict_test.values():
            m.update_state(y, out)
        i += 1

    # 输出测试的
    info = '{0} Samples has been tested'.format(test_sample_num)
    for m in metrics:
        info = info + ', {0}: {1}'.format(m, metrics_dict_test[m].result()[0])
    print(info)
    # 输出结束
    tmp_test = {}
    for m in metrics:
        tmp_test[m] = metrics_dict_test[m].result()[0]

    return tmp_test


