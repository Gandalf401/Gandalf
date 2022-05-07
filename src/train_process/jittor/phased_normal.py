import logging
import math

import jittor

from src.losses.loss_by_name import loss_by_name
from src.optimizers.optimizer_by_name import optimizer_by_name
from src.metircs.metric_tables import get_jittor_metrics
from src.metircs.jittor_expansion.mean import Mean


def phased_normal_train(model, optimizer, loss, epochs, batch_size,
                 ds_train, ds_valid=None, metrics=['OneHotAccuracy'], shuffle=True, show_iter=False):
    # 生成metrics存储字典
    metrics_dict_train = {}
    metrics_dict_valid = {}
    for m in metrics:
        metrics_dict_train[m] = get_jittor_metrics(m)
        metrics_dict_valid[m] = get_jittor_metrics(m)
    loss_metric_train = Mean()
    loss_metric_valid = Mean()
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
    train_optim = optimizer_by_name(optimizer['name'], 'Jittor', optim_params, model.parameters())
    logging.info('[JsonDL] Optimizer {0} has been defined.'.format(optimizer['name']))
    # 获取loss
    train_loss = loss_by_name(loss, 'Jittor')
    logging.info('[JsonDL] Loss {0} has been defined.'.format(loss))
    logging.info('[JsonDL] Preparing for your dataset, it may take some time...')
    # 处理数据集
    ds_train = ds_train.set_attrs(batch_size=batch_size, shuffle=shuffle, drop_last=False)
    batch_num_train = len(ds_train)
    logging.info('[JsonDL] Dataset has been ready.')
    # 验证集预处理
    if ds_valid is not None:
        # 不分batch 不shuffle
        batch_num_valid = len(ds_valid)

    # 开始训练
    for epoch in range(epochs):
        # 训练步骤
        i = 0
        for x, y in ds_train:
            if show_iter:
                print('Epoch={0}, Batch={1}/{2}, training...'.format(epoch + 1, i + 1, batch_num_train))
            out = model(x)
            if y.size(-1) != 1:
                # jittor的argmax函数返回包含两个张量的列表
                y_no_one_hot = y.argmax(1)[0]
            else:
                y_no_one_hot = y
            loss_value = train_loss(out, y_no_one_hot)
            # 后面三句相当于train_optim.step(loss_value)
            train_optim.zero_grad()
            train_optim.backward(loss_value)
            train_optim.step()
            # metrics
            # loss_metric_train.update_state(loss_value)
            # for m in metrics_dict_train.values():
            #     m.update_state(y, out)
            i += 1

        # 验证步骤
        if ds_valid is not None:
            i = 0
            for x, y in ds_valid:
                if show_iter:
                    print('Epoch={0}, Sample={1}/{2}, validating...'.format(epoch, i + 1, batch_num_valid))
                out = model(x)
                loss_value = loss(out, y)
                # metrics
                loss_metric_valid.update_state(loss_value)
                for m in metrics_dict_valid.values():
                    m.update_state(y, out)
                i += 1

        # 输出每epoch的信息
        info = [('Loss', loss_metric_train.result())]
        if ds_valid is not None:
            valid_info = [('Loss', loss_metric_valid.result())]
        else:
            valid_info = []
        for m in metrics:
            info.append((m, metrics_dict_train[m].result()))
            if ds_valid is not None:
                valid_info.append((m, metrics_dict_valid[m].result()))

        yield info, valid_info, epoch == epochs - 1


        # 每个epoch结束都清空状态
        loss_metric_train.reset_states()
        loss_metric_valid.reset_states()
        for m in metrics:
            metrics_dict_train[m].reset_states()
            metrics_dict_valid[m].reset_states()
