import logging

import tensorflow as tf

from src.losses.loss_by_name import loss_by_name
from src.optimizers.optimizer_by_name import optimizer_by_name
from src.metircs.metric_tables import get_tf_metrics


def normal_train(model, optimizer, loss, epochs, batch_size,
                 ds_train, ds_valid=None, metrics=['OneHotAccuracy'], localized=False, shuffle=True):
    tf.config.experimental_run_functions_eagerly(True)
    if not localized:
        # 初始化metrics records
        metrics_records_train = []
        metrics_records_valid = []
    # 生成metrics存储字典
    metrics_dict_train = {}
    metrics_dict_valid = {}
    for m in metrics:
        metrics_dict_train[m] = get_tf_metrics(m)
        metrics_dict_valid[m] = get_tf_metrics(m)
    loss_metric_train = tf.keras.metrics.Mean(name='train_loss_per_epoch')
    loss_metric_valid = tf.keras.metrics.Mean(name='valid_loss_per_epoch')
    # 根据optimizer参数进行配置
    if isinstance(optimizer, str):
        optimizer = {
            'name': optimizer
        }
    elif not isinstance(optimizer, dict):
        raise Exception('Param \"optimizer\" should be a str or dict.')
    # 生成新的optimizer
    if optimizer.__contains__('params'):
        optim_params = optimizer['params']
    else:
        optim_params = {}
    train_optim = optimizer_by_name(optimizer['name'], 'TensorFlow', optim_params)
    logging.info('[JsonDL] Optimizer {0} has been defined.'.format(optimizer['name']))
    # 获取loss
    train_loss = loss_by_name(loss, 'TensorFlow')
    logging.info('[JsonDL] Loss {0} has been defined.'.format(loss))
    logging.info('[JsonDL] Preparing for your dataset, it may take some time...')
    # 开始训练
    # shuffle -> 分batch
    if shuffle:
        ds_train = ds_train.shuffle(10)
    ds_train = ds_train.batch(batch_size)
    batch_num_train = sum(1 for _ in ds_train)
    ds_train = ds_train.repeat()
    it_train = tf.compat.v1.data.make_one_shot_iterator(ds_train)
    logging.info('[JsonDL] Dataset has been ready.')
    # 验证集预处理
    if ds_valid is not None:
        # 不分batch 不shuffle
        ds_valid = ds_valid.batch(batch_size)
        batch_num_valid = sum(1 for _ in ds_valid)
    # 开始训练
    for epoch in range(epochs):

        # 训练步骤
        for i in range(batch_num_train):
            print('Epoch={0}, Batch={1}/{2}, training...'.format(epoch + 1, i + 1, batch_num_train))
            x, y = it_train.get_next()
            train_step(model, x, y, train_loss, train_optim, loss_metric_train, metrics_dict_train)

        # 验证步骤
        if ds_valid is not None:
            it_valid = tf.compat.v1.data.make_one_shot_iterator(ds_valid)
            for i in range(batch_num_valid):
                print('Epoch={0}, Sample={1}/{2}, validating...'.format(epoch + 1, i + 1, batch_num_valid))
                x, y = it_valid.get_next()
                valid_step(model, x, y, train_loss, loss_metric_valid, metrics_dict_valid)

        # 输出每epoch的信息
        info = 'Epoch={0}, Loss:{1}'.format(epoch + 1, loss_metric_train.result())
        if ds_valid is not None:
            valid_info = '\t--Valid process, Loss:{0}'.format(loss_metric_valid.result())
        for m in metrics:
            info = info + ', {0}:{1}'.format(m, metrics_dict_train[m].result())
            if ds_valid is not None:
                valid_info = valid_info + ', {0}:{1}'.format(m, metrics_dict_valid[m].result())
        print(info)
        if ds_valid is not None:
            print(valid_info)
        # 输出结束 根据用户配置 选择内存存储或本地存储
        tmp_train = {}
        tmp_valid = {}
        for m in metrics:
            tmp_train[m] = metrics_dict_train[m].result().numpy()
            tmp_valid[m] = metrics_dict_valid[m].result().numpy()
        # 加上loss项
        tmp_train['Loss'] = loss_metric_train.result().numpy()
        tmp_valid['Loss'] = loss_metric_valid.result().numpy()
        # 内存存储
        if not localized:
            metrics_records_train.append(dict(tmp_train))
            metrics_records_valid.append(dict(tmp_valid))
        else:
            # 本地存储
            with open('./train_metrics', 'a+') as f:
                f.write(str(tmp_train) + '\n')
            with open('./valid_metrics', 'a+') as f:
                f.write(str(tmp_valid) + '\n')

        # 每个epoch结束都清空状态
        loss_metric_train.reset_states()
        loss_metric_valid.reset_states()
        for m in metrics:
            metrics_dict_train[m].reset_states()
            metrics_dict_valid[m].reset_states()

    # 所有epoch结束
    if localized:
        return './train_metrics', './valid_metrics'
    else:
        return metrics_records_train, metrics_records_valid


@tf.function
def train_step(model, x, y, loss, optim, loss_metric_train, metrics):
    with tf.GradientTape() as tape:
        predicts = model(x, training=True)
        loss_value = loss(y, predicts)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))
    # 根据metrics列表进行更新
    loss_metric_train.update_state(loss_value)
    for m in metrics.values():
        m.update_state(y, predicts)


@tf.function
def valid_step(model, x, y, loss, loss_metric_valid, metrics):
    predicts = model(x)
    loss_value = loss(y, predicts)
    # 根据metrics列表进行更新
    loss_metric_valid.update_state(loss_value)
    for m in metrics.values():
        m.update_state(y, predicts)
