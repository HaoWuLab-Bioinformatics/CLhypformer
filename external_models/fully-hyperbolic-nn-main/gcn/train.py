from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
from optim import RiemannianAdam, RiemannianSGD
import os
import pickle
import time
from pathlib import Path

# 定义项目路径
gcn_dir = Path(__file__).parent

# 如果环境变量未设置，使用默认路径
if 'DATAPATH' not in os.environ:
    os.environ['DATAPATH'] = str(gcn_dir / 'data' / 'raw')

if 'LOG_DIR' not in os.environ:
    os.environ['LOG_DIR'] = str(gcn_dir / 'logs')

import numpy as np
import torch
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics

from geoopt_compat import ManifoldParameter


def calculate_class_weights(labels, idx_train, n_classes):
    """动态计算训练集中的类别权重"""
    train_labels = labels[idx_train]
    if torch.is_tensor(train_labels):
        train_labels = train_labels.cpu().numpy()

    class_counts = np.bincount(train_labels, minlength=n_classes)
    class_counts[class_counts == 0] = 1
    weights = 1.0 / np.sqrt(class_counts)
    weights = weights / np.sum(weights) * n_classes

    return torch.FloatTensor(weights)


def train(args, data=None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else int(args.patience)
    logging.getLogger().setLevel(logging.INFO)

    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ.get('LOG_DIR', 'logs'), args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(
                                    os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])
    else:
        save_dir = None

    logging.info(f'Using: {args.device}')

    datapath = os.environ.get('DATAPATH', os.path.join(gcn_dir, 'data', 'raw'))
    if data is None:
        data = load_data(args, os.path.join(datapath, args.dataset))

    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        args.data = data
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    model = Model(args)
    logging.info(str(model))

    if args.task == 'nc':
        class_weights = calculate_class_weights(data['labels'], data['idx_train'], args.n_classes)
        class_weights = class_weights.to(args.device)
        model.weights = class_weights

        loss_injected = False
        for module in model.modules():
            if isinstance(module, torch.nn.CrossEntropyLoss):
                module.weight = class_weights
                loss_injected = True

    no_decay = ['bias', 'scale']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if p.requires_grad and not any(
                nd in n
                for nd in no_decay) and not isinstance(p, ManifoldParameter)
        ],
        'weight_decay': args.weight_decay
    }, {
        'params': [
            p for n, p in model.named_parameters() if p.requires_grad and (any(
                nd in n
                for nd in no_decay) or isinstance(p, ManifoldParameter))
        ],
        'weight_decay': 0.0
    }]

    if args.optimizer == 'radam':
        optimizer = RiemannianAdam(params=optimizer_grouped_parameters, lr=args.lr, stabilize=1)
    elif args.optimizer == 'rsgd':
        optimizer = RiemannianSGD(params=optimizer_grouped_parameters, lr=args.lr, stabilize=1)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.lr_reduce_freq),
                                                   gamma=float(args.gamma))
    model = model.to(args.device)

    # === 将鲁棒性生成的重算图转移到对应的 GPU/CPU ===
    for x, val in data.items():
        if torch.is_tensor(data[x]):
            data[x] = data[x].to(args.device)

    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()

        grad_clip = args.grad_clip if (args.grad_clip is not None and args.grad_clip > 0) else 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        lr_scheduler.step()

        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join([
                'Epoch: {:04d}'.format(epoch + 1),
                'lr: {}'.format(lr_scheduler.get_last_lr()),
                format_metrics(train_metrics, 'train'),
                'time: {:.4f}s'.format(time.time() - t)
            ]))

        with torch.no_grad():
            if (epoch + 1) % args.eval_freq == 0:
                model.eval()
                embeddings = model.encode(data['features'], data['adj_train_norm'])
                val_metrics = model.compute_metrics(embeddings, data, 'val')

                if (epoch + 1) % args.log_freq == 0:
                    logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))

                if model.has_improved(best_val_metrics, val_metrics):
                    best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                    best_emb = embeddings.cpu()
                    if args.save and save_dir:
                        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                        np.save(os.path.join(save_dir, 'labels.npy'), data['labels'].cpu().detach().numpy())
                    best_val_metrics = val_metrics
                    counter = 0
                else:
                    counter += 1
                    if counter == args.patience and epoch > args.min_epochs:
                        logging.info("Early stopping")
                        break

    logging.info("Optimization Finished!")
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test')

    if args.save and save_dir:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        np.save(os.path.join(save_dir, 'labels.npy'), data['labels'].cpu().detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))

        config_dict = vars(args).copy()
        config_dict.pop('data', None)
        for key, value in config_dict.items():
            if hasattr(value, 'item'):
                config_dict[key] = value.item()
            elif hasattr(value, 'tolist'):
                config_dict[key] = value.tolist()

        json.dump(config_dict, open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))

    # == 增强版字典浮点转换 ==
    def convert_metrics_to_float(metrics):
        if metrics is None:
            return {}
        safe_metrics = {}
        for k, v in metrics.items():
            if torch.is_tensor(v):
                safe_metrics[k] = float(v.cpu().item())
            else:
                safe_metrics[k] = float(v)
        return safe_metrics

    return convert_metrics_to_float(best_val_metrics), convert_metrics_to_float(best_test_metrics)