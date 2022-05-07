import logging

import tensorflow as tf

from src.metircs.metric_tables import get_tf_metrics


def normal_test(model, ds_test, metrics=['OneHotAccuracy']):
    # 生成metrics存储字典
    metrics_dict_test = {}
    for m in metrics:
        metrics_dict_test[m] = get_tf_metrics(m)
    logging.info('[JsonDL] Preparing for your dataset, it may take some time...')
    # 准备数据集
    ds_test = ds_test.batch(1)
    test_sample_num = sum(1 for _ in ds_test)
    it_test = tf.compat.v1.data.make_one_shot_iterator(ds_test)
    logging.info('[JsonDL] Dataset has been ready.')
    # 开始测试
    for i in range(test_sample_num):
        print('Sample={0}/{1}, testing...'.format(i + 1, test_sample_num))
        x, y = it_test.get_next()
        test_step(model, x, y, metrics_dict_test)

    # 输出测试的
    info = '{0} Samples has been tested'.format(test_sample_num)
    for m in metrics:
        info = info + ', {0}:{1}'.format(m, metrics_dict_test[m].result())
    print(info)
    # 输出结束
    tmp_test = {}
    for m in metrics:
        tmp_test[m] = metrics_dict_test[m].result().numpy()

    return tmp_test


@tf.function
def test_step(model, x, y, metrics):
    predicts = model(x, training=False)
    for m in metrics.values():
        m.update_state(y, predicts)
