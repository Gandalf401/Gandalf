import logging

import torch
from torch.utils.data import DataLoader

from src.losses.loss_by_name import loss_by_name
from src.optimizers.optimizer_by_name import optimizer_by_name
from src.metircs.metric_tables import get_torch_metrics
from src.metircs.torch_expansion.mean import Mean


def phased_normal_train(model, optimizer, loss, epochs, batch_size,
                 ds_train, ds_valid=None, metrics=['OneHotAccuracy'], shuffle=True, show_iter=False):
    # 确保使用了gpu
    gpu = torch.device("cuda:0")

    # 生成metrics存储字典
    metrics_dict_train = {}
    metrics_dict_valid = {}
    for m in metrics:
        metrics_dict_train[m] = get_torch_metrics(m)
        metrics_dict_valid[m] = get_torch_metrics(m)
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
    train_optim = optimizer_by_name(optimizer['name'], 'PyTorch', optim_params, model.parameters())
    logging.info('[JsonDL] Optimizer {0} has been defined.'.format(optimizer['name']))
    # 获取loss
    train_loss = loss_by_name(loss, 'PyTorch')
    logging.info('[JsonDL] Loss {0} has been defined.'.format(loss))
    logging.info('[JsonDL] Preparing for your dataset, it may take some time...')
    # 处理数据集
    ds_train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle)
    batch_num_train = sum(1 for _ in ds_train_loader)
    logging.info('[JsonDL] Dataset has been ready.')
    # 验证集预处理
    if ds_valid is not None:
        # 不分batch 不shuffle
        batch_num_valid = sum(1 for _ in ds_valid)
        ds_valid_loader = DataLoader(ds_valid, 1, False)

    # 开始训练
    model.train()
    for epoch in range(epochs):
        # 训练步骤
        i = 0
        for x, y in ds_train_loader:
            if show_iter:
                print('Epoch={0}, Batch={1}/{2}, training...'.format(epoch + 1, i + 1, batch_num_train))
            # batch_x, batch_y = torch.Variable(x), torch.Variable(y)
            x = x.to(gpu)
            y = y.to(gpu)
            x = x.to(torch.float32)
            y = y.to(torch.float32)
            x = torch.autograd.Variable(x)
            y = torch.autograd.Variable(y)
            out = model(x)
            # 判断一下y是不是one-hot
            if y.size(-1) != 1:
                y_no_one_hot = y.argmax(1)
            else:
                 y_no_one_hot = y
            loss_value = train_loss(out, y_no_one_hot)
            train_optim.zero_grad()
            loss_value.backward()
            train_optim.step()
            # metrics
            loss_metric_train.update_state(loss_value)
            for m in metrics_dict_train.values():
                m.update_state(y, out)
            i += 1

        # 验证步骤
        if ds_valid is not None:
            i = 0
            for x, y in ds_valid_loader:
                if show_iter:
                    print('Epoch={0}, Sample={1}/{2}, validating...'.format(epoch, i + 1, batch_num_valid))
                # batch_x, batch_y = torch.Variable(x), torch.Variable(y)
                x = x.to(torch.float32)
                y = y.to(torch.float32)
                x = torch.autograd.Variable(x)
                y = torch.autograd.Variable(y)
                out = model(x)
                loss_value = train_loss(out, y)
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
