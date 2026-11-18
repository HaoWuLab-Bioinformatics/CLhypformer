import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import class_rand_splits, eval_acc, evaluate, load_fixed_splits
from dataset import load_nc_dataset
from logger import Logger
from parse import parse_method, parser_add_main_args
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

from manifolds.hyp_layer import Optimizer
from topology_analysis import save_topology_comparison, save_topology_snapshot

warnings.filterwarnings('ignore')


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def measure_cl_gradients(task_loss, cl_branch_loss, model):
    """Measure competing gradients on the encoder shared by both objectives."""
    params = tuple(p for p in (
        tuple(model.adv_module.encoder_l1.parameters())
        + tuple(model.adv_module.encoder_l2.parameters())) if p.requires_grad)
    task_grad = torch.autograd.grad(task_loss, params, retain_graph=True,
                                    allow_unused=True)
    cl_grad = torch.autograd.grad(cl_branch_loss, params, retain_graph=True,
                                  allow_unused=True)
    task_sq = sum((g.detach() ** 2).sum() for g in task_grad if g is not None)
    cl_sq = sum((g.detach() ** 2).sum() for g in cl_grad if g is not None)
    dot = sum((a.detach() * b.detach()).sum()
              for a, b in zip(task_grad, cl_grad) if a is not None and b is not None)
    task_norm = float(task_sq.sqrt()) if isinstance(task_sq, torch.Tensor) else 0.0
    cl_norm = float(cl_sq.sqrt()) if isinstance(cl_sq, torch.Tensor) else 0.0
    return {'task_grad_norm': task_norm, 'cl_grad_norm': cl_norm,
            'cl_to_task_grad_ratio': cl_norm / max(task_norm, 1e-12),
            'grad_cosine': float(dot) / max(task_norm * cl_norm, 1e-12)}


def pretrain_contrastive_module(model, dataset, args, device, train_idx=None):
    """
    预训练对比学习组件
    专门针对 adv_module (Encoder, 投影头, 视图增强判别器, InfoNCE损失) 及曲率参数进行预训练
    """
    print("🚀 开始对比学习模块(CL)的预训练...")
    model.train()

    pretrain_params = [
        {'params': model.adv_module.parameters(), 'weight_decay': getattr(args, 'weight_decay', 1e-4)}
    ]

    if getattr(args, 'use_adaptive_curvature', False):
        pretrain_params.append({'params': model.k_in_learner.parameters(), 'weight_decay': 0.0})
        pretrain_params.append({'params': model.k_out_learner.parameters(), 'weight_decay': 0.0})

    try:
        from geoopt.optim import RiemannianAdam
        optimizer = RiemannianAdam(pretrain_params, lr=args.lr)
    except ImportError:
        import torch.optim as optim
        optimizer = optim.Adam(pretrain_params, lr=args.lr)

    x = dataset.graph['node_feat'].to(device)
    epochs = getattr(args, 'pretrain_epochs', 50)

    for epoch in range(epochs):
        optimizer.zero_grad()
        model._update_manifolds()

        objective = 'supcon' if args.cl_mode == 'shared_supcon' else 'legacy'
        cl_labels = (dataset.cl_fine_labels if args.cl_fine_labels
                     else dataset.label.squeeze(1))
        z1_hyp, z2_hyp, contrast_loss, _ = model.adv_module(
            x, labels=cl_labels if objective == 'supcon' else None,
            train_idx=train_idx, objective=objective,
            group_ids=dataset.patient_codes if args.cl_cross_patient else None,
            cross_group=bool(args.cl_cross_patient),
            coarse_labels=dataset.label.squeeze(1),
            mask_same_major=bool(args.cl_mask_same_major),
            pair_policy=args.cl_pair_policy,
            patient_balance=bool(args.cl_patient_balance))

        contrast_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [Pretrain] Epoch {epoch + 1:03d}/{epochs} | Contrastive Loss: {contrast_loss.item():.4f}")

    print("✅ 对比学习模块预训练完成！\n" + "=" * 40)
    torch.cuda.empty_cache()
    return model


def drop_edges(edge_index, drop_rate):
    if drop_rate <= 0:
        return edge_index
    num_edges = edge_index.size(1)
    num_keep = int(num_edges * (1.0 - drop_rate))

    perm = torch.randperm(num_edges, device=edge_index.device)
    keep_indices = perm[:num_keep]
    return edge_index[:, keep_indices]


def mask_features(node_feat, drop_rate):
    if drop_rate <= 0:
        return node_feat
    mask = (torch.rand_like(node_feat) > drop_rate).to(node_feat.dtype)
    return node_feat * mask


def run_robustness_test(model, dataset, split_idx, device, seed, trials=3):
    """Paired, repeatable edge and feature corruption on the held-out test set."""
    model.eval()
    drop_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
    test_idx = split_idx['test'].to(device)
    labels = dataset.label.squeeze(1)[test_idx]
    orig_edge = dataset.graph['edge_index']
    orig_edge_tensor = orig_edge[0] if isinstance(orig_edge, tuple) else orig_edge
    orig_feat = dataset.graph['node_feat']
    num_nodes = orig_feat.size(0)
    # Group reverse directions so each undirected connection is dropped together.
    edge_keys = (torch.minimum(orig_edge_tensor[0], orig_edge_tensor[1]) * num_nodes
                 + torch.maximum(orig_edge_tensor[0], orig_edge_tensor[1]))
    unique_keys, inverse = torch.unique(edge_keys, return_inverse=True)
    self_loops = unique_keys.div(num_nodes, rounding_mode='floor') == unique_keys.remainder(num_nodes)
    results = {'edge': {}, 'feat': {}, 'edge_f1': {}, 'feat_f1': {},
               'edge_trials': {}, 'feat_trials': {}}
    adv_module = getattr(model, 'adv_module', None)
    if adv_module is not None and getattr(model, 'use_adv', False):
        # The baseline uses its original graph when features are masked.
        # Hold the CL graph from clean features fixed under the same condition.
        adv_module.cached_eval_h = None
        adv_module.freeze_eval_graph = True
    try:
        with torch.no_grad():
            for kind in ('edge', 'feat'):
                for rate in drop_rates:
                    scores = []
                    f1_scores = []
                    for trial in range(trials):
                        generator = torch.Generator(device=device)
                        generator.manual_seed(int(seed) + 1009 * trial + (0 if kind == 'edge' else 104729))
                        if kind == 'edge' and rate:
                            keep = (torch.rand(unique_keys.numel(), device=device, generator=generator) >= rate)
                            keep = keep | self_loops
                            new_edge = orig_edge_tensor[:, keep[inverse]]
                            dataset.graph['edge_index'] = (new_edge,) if isinstance(orig_edge, tuple) else new_edge
                        elif kind == 'feat' and rate:
                            mask = torch.rand(orig_feat.shape, device=device, generator=generator) >= rate
                            dataset.graph['node_feat'] = orig_feat * mask
                        output = model(dataset)
                        if isinstance(output, tuple):
                            output = output[0]
                        pred = output[test_idx].argmax(dim=1)
                        scores.append(100 * pred.eq(labels).float().mean().item())
                        f1_scores.append(f1_score(
                            labels.cpu(), pred.cpu(), labels=np.arange(output.size(-1)),
                            average='macro', zero_division=0))
                        dataset.graph['edge_index'] = orig_edge
                        dataset.graph['node_feat'] = orig_feat
                    results[kind][rate] = float(np.mean(scores))
                    results[kind + '_f1'][rate] = float(np.mean(f1_scores))
                    results[kind + '_trials'][rate] = scores
                    print(f'Robustness {kind} {rate:.1f}: acc={np.mean(scores):.2f}% '
                          f'macro_f1={np.mean(f1_scores):.4f} trials={trials}')
    finally:
        dataset.graph['edge_index'] = orig_edge
        dataset.graph['node_feat'] = orig_feat
        if adv_module is not None:
            adv_module.freeze_eval_graph = False
            adv_module.cached_eval_h = None
    return results


def calculate_f1(model, dataset, split_idx, device):
    """
    计算模型在train/val/test上的F1分数
    """
    model.eval()

    with torch.no_grad():
        out = model(dataset)
        if isinstance(out, tuple): out = out[0]

        train_out = out[split_idx['train'].to(device)]
        val_out = out[split_idx['valid'].to(device)]
        test_out = out[split_idx['test'].to(device)]

        train_pred = train_out.max(1)[1]
        val_pred = val_out.max(1)[1]
        test_pred = test_out.max(1)[1]

        train_labels = dataset.label.squeeze(1)[split_idx['train']].to(device)
        val_labels = dataset.label.squeeze(1)[split_idx['valid']].to(device)
        test_labels = dataset.label.squeeze(1)[split_idx['test']].to(device)

        train_f1 = f1_score(train_labels.cpu(), train_pred.cpu(), average='weighted', zero_division=0)
        val_f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=0)
        test_f1 = f1_score(test_labels.cpu(), test_pred.cpu(), average='weighted', zero_division=0)

    return train_f1, val_f1, test_f1


def validation_class_metrics(model, dataset, valid_idx, device, num_classes):
    """Score a checkpoint using validation labels only, including rare-class recall."""
    model.eval()
    with torch.no_grad():
        logits = model(dataset)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = logits[valid_idx.to(device)].argmax(dim=-1).cpu().numpy()
    truth = dataset.label.squeeze(1)[valid_idx].cpu().numpy()
    present = np.unique(truth)
    labels = np.arange(num_classes)
    return {
        'macro_f1': float(f1_score(truth, pred, labels=present,
                                   average='macro', zero_division=0)),
        'class_recall': recall_score(truth, pred, labels=labels,
                                     average=None, zero_division=0).tolist(),
        'class_support': np.bincount(truth, minlength=num_classes).tolist(),
    }


### Parse args ###
parser = argparse.ArgumentParser(description='Medium Data Training Pipeline')
parser.add_argument('--save_topology', type=int, default=1,
                    help='save topology evolution snapshots (H, embeddings, metrics)')
parser.add_argument('--topology_dir', type=str, default='results/topology',
                    help='output directory for topology snapshots')
parser.add_argument('--topology_epoch', type=int, default=-1,
                    help='0-based epoch to snapshot; -1 uses the best epoch')
parser.add_argument('--topology_csv', type=int, default=1,
                    help='also export sorted incidence matrices as CSV')
parser.add_argument('--topology_plot', type=int, default=1,
                    help='also render sorted incidence matrix heatmaps')
parser_add_main_args(parser)
parser.add_argument('--use_cl', nargs='?', type=int, const=1, default=0, choices=[0, 1],
                    help='是否使用超图对比学习 (0 关闭, 1 开启)')
parser.add_argument('--no_use_cl', action='store_false', dest='use_cl', help='禁用超图对比学习')
parser.add_argument('--cl_weight', type=float, default=0.1, help='对比损失(contrast_loss)的权重')
parser.add_argument('--cl_mode', type=str, default='shared_supcon',
                    choices=['shared_supcon', 'patient_prototype', 'legacy'],
                    help='共享对比、患者原型或旧版损失')
parser.add_argument('--hyperedge_policy',
                    choices=['global_learned', 'patient_state'],
                    default='global_learned',
                    help='global learned kNN or patient-local multiscale cell-state hyperedges')
parser.add_argument('--prototype_temperature', type=float, default=0.2,
                    help='temperature for leave-one-patient-out class prototypes')
parser.add_argument('--prototype_pcgrad', type=int, choices=[0, 1], default=0,
                    help='project conflicting prototype gradients off task gradients')
parser.add_argument('--cl_aux_weight', type=float, default=0.1,
                    help='supervised loss weight for the two contrastive views')
parser.add_argument('--cl_fusion_weight', type=float, default=0.0,
                    help='residual contribution of learned contrastive features to task input')
parser.add_argument('--cl_knn', type=int, default=5, help='构建KNN对比图的邻居数K')
parser.add_argument('--cl_drop_rate', type=float, default=0.2, help='增强视图随机丢弃边的概率')
parser.add_argument('--cl_feature_noise', type=float, default=0.05,
                    help='第二个对比视图中的切空间高斯噪声尺度')
parser.add_argument('--cl_aux_head', type=int, choices=[0, 1], default=1,
                    help='辅助分类头消融开关')
parser.add_argument('--cl_consistency', type=int, choices=[0, 1], default=1,
                    help='扰动一致性约束消融开关')
parser.add_argument('--cl_consistency_weight', type=float, default=0.2)
parser.add_argument('--cl_fine_labels', type=int, choices=[0, 1], default=0,
                    help='use minor cell types as contrastive positives; main classifier remains major type')
parser.add_argument('--cl_cross_patient', type=int, choices=[0, 1], default=0,
                    help='prefer same-type positives from different patients')
parser.add_argument('--cl_pair_policy', choices=['diag_fixed', 'legacy'],
                    default='diag_fixed', help='whether paired views remain positives in cross-patient CL')
parser.add_argument('--cl_mask_same_major', type=int, choices=[0, 1], default=0,
                    help='exclude same-major different-minor pairs from CL negatives')
parser.add_argument('--cl_patient_balance', type=int, choices=[0, 1], default=0,
                    help='give each positive patient equal weight per anchor')
parser.add_argument('--cl_warmup_epochs', type=int, default=0,
                    help='supervised epochs with zero CL weight')
parser.add_argument('--cl_ramp_epochs', type=int, default=0,
                    help='supervised epochs to reach the configured CL weight after warmup')
parser.add_argument('--cl_grad_diagnostics', type=int, choices=[0, 1], default=0,
                    help='measure task and CL gradient norms on shared encoder')
parser.add_argument('--cl_grad_file', type=str, default='',
                    help='optional JSONL path for gradient diagnostics')
parser.add_argument('--robustness_file', type=str, default='',
                    help='save paired robustness metrics as JSON')
parser.add_argument('--cl_tau', type=float, default=0.5, help='双曲对比损失的温度系数')
parser.add_argument('--do_pretrain', action='store_true', default=True, help='是否开启对比学习预训练')
parser.add_argument('--no-pretrain', action='store_false', dest='do_pretrain',
                    help='disable contrastive pretraining for a clean loss ablation')
parser.add_argument('--pretrain_epochs', type=int, default=50, help='对比学习预训练的轮数')
parser.add_argument('--use_adaptive_curvature', action='store_true', default=False,
                    help='是否开启自适应曲率 (Adaptive Curvature)')
parser.add_argument('--knn_metric', type=str, default='hyp', choices=['hyp', 'euc'],
                    help='构建KNN图的度量空间 (hyp或euc)')
parser.add_argument('--cl_space', type=str, default='hyp', choices=['hyp', 'euc'], help='对比学习所处空间 (hyp或euc)')
parser.add_argument('--graph_build_method', type=str, default='knn', choices=['knn', 'random_walk'],
                    help='图构建方式: knn或距离阈值eps或随机游走random_walk')
parser.add_argument('--drop_strategy', type=str, default='random', choices=['critical', 'random', 'topk'],
                    help='视图增强扰动策略')
parser.add_argument('--walk_length', type=int, default=3, help='随机游走步长')
parser.add_argument('--graph_source', type=str, default='learned', choices=['input', 'learned'],
                    help='build the contrastive hypergraph from raw input or learned embeddings')
parser.add_argument('--topology_consensus', type=int, default=0,
                    help='average this many stochastic passes for evolved snapshots (0 disables)')
parser.add_argument('--topology_class_block', type=int, default=1,
                    help='also render class-block density heatmaps')
parser.add_argument('--topology_label_loss', type=int, default=1,
                    help='supervise embedding similarity with labeled pairs')
parser.add_argument('--topology_label_weight', type=float, default=0.1,
                    help='weight of the label-guided topology loss')
parser.add_argument('--prediction_file', type=str, default='',
                    help='optional NPZ file for test predictions at the best validation epoch')
parser.add_argument('--checkpoint_file', type=str, default='',
                    help='optional checkpoint selected by the configured validation metric')
parser.add_argument('--checkpoint_metric', choices=['accuracy', 'macro_f1'],
                    default='accuracy', help='validation metric used to select a checkpoint')
parser.add_argument('--min_checkpoint_epoch', type=int, default=1,
                    help='earliest 1-based training epoch eligible as a checkpoint')
parser.add_argument('--embedding_file', type=str, default='',
                    help='optional all-node pre-classifier representations at the selected checkpoint')
args = parser.parse_args()
if (args.cl_fine_labels or args.cl_cross_patient) and args.dataset != 'gse176078_brca':
    parser.error('fine cell labels and patient-aware positives require GSE176078')
if args.cl_mask_same_major and not args.cl_fine_labels:
    parser.error('same-major negative masking requires fine labels')
if args.cl_patient_balance and not args.cl_cross_patient:
    parser.error('patient-balanced positives require cross-patient CL')
if args.cl_warmup_epochs < 0 or args.cl_ramp_epochs < 0:
    parser.error('CL warmup and ramp lengths must be nonnegative')
if args.min_checkpoint_epoch < 1 or args.min_checkpoint_epoch > args.epochs:
    parser.error('min_checkpoint_epoch must be between 1 and epochs')
if args.cl_mode == 'patient_prototype':
    if args.dataset != 'gse176078_brca':
        parser.error('patient prototypes require GSE176078 patient IDs')
    if args.do_pretrain:
        parser.error('patient prototypes require --no-pretrain')
    if args.prototype_temperature <= 0:
        parser.error('prototype_temperature must be positive')
if args.hyperedge_policy == 'patient_state' and (
        args.dataset != 'gse176078_brca' or not args.use_cl or
        args.cl_mode != 'patient_prototype'):
    parser.error('patient-state hyperedges require GSE176078 patient_prototype fusion')
if args.prototype_pcgrad and args.cl_mode != 'patient_prototype':
    parser.error('prototype_pcgrad requires patient_prototype mode')
if args.use_cl and args.cl_mode == 'shared_supcon' and args.drop_strategy != 'random':
    parser.error('shared_supcon currently requires --drop_strategy random; use legacy for old strategies')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
print('====' * 20)
print(args)

if args.method in ('gcn', 'gat', 'mlp'):
    args.use_cl = False
    args.do_pretrain = False
    args.use_adaptive_curvature = False

if args.save_topology:
    experiment_tag = 'clhyperformer' if args.use_cl else 'hyperformer'
    topology_experiment_dir = os.path.join(
        args.topology_dir,
        f'{experiment_tag}_{time.strftime("%Y%m%d_%H%M%S")}',
    )
    topology_params = {
        'dataset': args.dataset,
        'method': args.method,
        'use_cl': bool(args.use_cl),
        'graph_build_method': args.graph_build_method,
        'graph_source': args.graph_source,
        'drop_strategy': args.drop_strategy,
        'knn_metric': args.knn_metric,
        'cl_drop_rate': args.cl_drop_rate,
        'topology_label_loss': args.topology_label_loss,
        'topology_label_weight': args.topology_label_weight,
        'topology_consensus': args.topology_consensus,
        'epochs': args.epochs,
        'patience': args.patience,
    }
else:
    topology_experiment_dir = args.topology_dir
    topology_params = None

if args.cpu:
    device = torch.device("cpu")
    print('>> Using CPU')
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print('>> Using GPU: ' + str(args.device))

### Load and preprocess data ###
dataset = load_nc_dataset(args)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)
all_labels = dataset.label.squeeze(1)

dataset_name = args.dataset

# 【修改】针对 wiki-new 数据集强制关闭随机划分，避免消融实验中引入不公平的随机干扰
if args.dataset == 'wiki-new' and getattr(args, 'rand_split', False):
    print('>> 🚨 检测到 wiki-new 数据集，已自动关闭 rand_split，强制使用固定划分。')
    args.rand_split = False

bio_datasets = ('cancer', 'CCLE', 'tcga_pancan', 'pancreas_sc', 'tabula_muris_lung', 'gse176078_brca')
if args.dataset in bio_datasets and args.rand_split and args.rand_split_class:
    if args.runs != 1:
        raise ValueError('For biological datasets, use yunxing_medium.py --runs N so each run reloads seed-specific preprocessing')
    split_idx_lst = [dataset.split_idx]
    split = split_idx_lst[0]
    split_bytes = np.concatenate([
        split[key].cpu().numpy().astype(np.int64)
        for key in ('train', 'valid', 'test')
    ]).tobytes()
    print(f'>> seeded class-stratified split: seed={args.seed}, '
          f'train={len(split["train"])}, valid={len(split["valid"])}, '
          f'test={len(split["test"])}, sha256={hashlib.sha256(split_bytes).hexdigest()[:16]}')
elif args.rand_split:
    print('>> loading random splits ...')
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    print('>> loading random class splits ...')
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
else:
    print('>> loading fixed splits ...')
    split_idx_lst = load_fixed_splits(
        dataset, name=args.dataset, protocol=args.protocol)

if args.dataset in ('mini', '20news'):
    adj_knn = kneighbors_graph(dataset.graph['node_feat'], n_neighbors=args.knn_num, include_self=True)
    edge_index = torch.tensor(adj_knn.nonzero(), dtype=torch.long)
    dataset.graph['edge_index'] = edge_index

n = dataset.graph['num_nodes']
num_class = max(dataset.label.max().item() + 1, dataset.label.shape[1])
num_class = (int)(num_class)
node_feat_dim = dataset.graph['node_feat'].shape[1]
args.in_channels = node_feat_dim
args.out_channels = num_class

dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)

print(f">> num nodes {n} | num classes {num_class} | num node feats {node_feat_dim}")

from result_saver import ResultSaver

result_saver = ResultSaver(args.dataset)

if args.dataset in ('deezer-europe'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

eval_func = eval_acc

logger = Logger(args.runs, args)
all_robustness_results = []

for run in range(args.runs):
    print(f'🔥Run {run + 1}/{args.runs}')
    if len(split_idx_lst) == 1:
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model = parse_method(args, device)

    if getattr(args, 'use_cl', True) and getattr(args, 'do_pretrain', False):
        model = pretrain_contrastive_module(model, dataset, args, device, train_idx)

    if getattr(args, 'use_cl', True) and args.cl_mode == 'legacy':
        print("🧊 应用策略 A：冻结对比模块底层，仅解冻 encoder_l2 与 adapt_layer...")

        for param in model.adv_module.parameters():
            param.requires_grad = False

        for name, param in model.adv_module.named_parameters():
            if 'encoder_l2' in name:
                param.requires_grad = True

        if hasattr(model, 'adapt_layer'):
            for param in model.adapt_layer.parameters():
                param.requires_grad = True

    optimizer = Optimizer(model, args)

    best_val = 0
    best_selection_score = -1.0
    best_validation_class_metrics = None
    patience = 0
    best_model_state = None

    best_train_acc = 0
    best_test_acc_at_best_val = 0
    best_epoch = 0
    initial_topology_metrics = None
    fixed_topology_metrics = None

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        if getattr(args, 'use_cl', False) and args.cl_mode in ('shared_supcon', 'patient_prototype'):
            if epoch < args.cl_warmup_epochs:
                model.cl_weight = 0.0
            elif args.cl_ramp_epochs:
                progress = (epoch - args.cl_warmup_epochs + 1) / args.cl_ramp_epochs
                model.cl_weight = args.cl_weight * min(1.0, progress)
            else:
                model.cl_weight = args.cl_weight

        labels = dataset.label.squeeze(1)[train_idx]

        snapshot = None
        if getattr(args, 'use_cl', True):
            out_orig, out1, out2, contrast_loss = model(dataset, train_idx=train_idx)
            if args.save_topology and hasattr(model, 'adv_module'):
                snapshot = getattr(model.adv_module, 'last_snapshot', None)
            loss_orig = F.nll_loss(F.log_softmax(out_orig[train_idx], dim=-1), labels)

            if out1 is None:
                task_loss = loss_orig
            elif out2 is None:
                loss1 = F.nll_loss(F.log_softmax(out1[train_idx], dim=-1), labels)
                task_loss = loss_orig + args.cl_aux_weight * loss1
            else:
                loss1 = F.nll_loss(F.log_softmax(out1[train_idx], dim=-1), labels)
                loss2 = F.nll_loss(F.log_softmax(out2[train_idx], dim=-1), labels)
                task_loss = loss_orig + args.cl_aux_weight * (loss1 + loss2)

            loss = task_loss + contrast_loss
        else:
            out = model(dataset)
            if (args.save_topology and hasattr(model, 'adv_module')
                    and (epoch == 0 or (args.topology_epoch >= 0 and epoch == args.topology_epoch))):
                model.adv_module.train()
                with torch.no_grad():
                    _, _, _, snapshot = model.adv_module(
                        dataset.graph['node_feat'].to(device))
            loss = F.nll_loss(F.log_softmax(out[train_idx], dim=-1), labels)

        if snapshot is not None:
            if epoch == 0 and initial_topology_metrics is None:
                initial_topology_metrics = save_topology_snapshot(
                    snapshot=snapshot,
                    labels=all_labels,
                    output_dir=topology_experiment_dir,
                    dataset_name=args.dataset,
                    run=run,
                    epoch=epoch,
                    tag='initial',
                    export_csv=bool(args.topology_csv),
                    export_plot=bool(args.topology_plot),
                    export_class_block=bool(args.topology_class_block),
                    params=topology_params,
                    images_subdir='figures',
                )
            if (args.topology_epoch >= 0 and epoch == args.topology_epoch
                    and fixed_topology_metrics is None):
                fixed_topology_metrics = save_topology_snapshot(
                    snapshot=snapshot,
                    labels=all_labels,
                    output_dir=topology_experiment_dir,
                    dataset_name=args.dataset,
                    run=run,
                    epoch=epoch,
                    tag='fixed',
                    export_csv=bool(args.topology_csv),
                    export_plot=bool(args.topology_plot),
                    export_class_block=bool(args.topology_class_block),
                    params=topology_params,
                    images_subdir='figures',
                )

        if (getattr(args, 'topology_label_loss', 0)
                and args.cl_mode == 'legacy'
                and getattr(args, 'use_cl', True)
                and hasattr(model, 'adv_module')):
            topology_homophily_loss = model.adv_module.embedding_homophily_loss(
                dataset.graph['node_feat'], dataset.label.squeeze(1), train_idx)
            loss = loss + args.topology_label_weight * topology_homophily_loss

        if (args.cl_grad_diagnostics and getattr(args, 'use_cl', False)
                and args.cl_mode in ('shared_supcon', 'patient_prototype')
                and epoch % max(args.display_step, 1) == 0):
            if args.cl_mode == 'patient_prototype':
                from gse_patient_modules import prototype_gradient_diagnostics
                diagnostic = prototype_gradient_diagnostics(
                    task_loss, contrast_loss, model)
                diagnostic['prototype_stats'] = model.last_prototype_stats
            else:
                diagnostic = measure_cl_gradients(task_loss, contrast_loss, model)
            diagnostic.update({'epoch': epoch, 'cl_weight': model.cl_weight,
                               'task_loss': float(task_loss.detach()),
                               'cl_branch_loss': float(contrast_loss.detach())})
            print('CL gradients: ' + json.dumps(diagnostic), flush=True)
            if args.cl_grad_file:
                os.makedirs(os.path.dirname(os.path.abspath(args.cl_grad_file)), exist_ok=True)
                with open(args.cl_grad_file, 'a', encoding='utf-8') as stream:
                    stream.write(json.dumps(diagnostic) + '\n')

        if (args.prototype_pcgrad and getattr(args, 'use_cl', False)
                and args.cl_mode == 'patient_prototype'):
            from gse_patient_modules import apply_prototype_pcgrad
            pcgrad_status = apply_prototype_pcgrad(task_loss, contrast_loss, model)
            if epoch % max(args.display_step, 1) == 0:
                print('Prototype PCGrad: ' + json.dumps(pcgrad_status), flush=True)
        else:
            loss.backward()
        optimizer.step()

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)

        logger.add_result(run, result[:-1])

        current_validation_class_metrics = None
        if args.checkpoint_metric == 'macro_f1':
            current_validation_class_metrics = validation_class_metrics(
                model, dataset, split_idx['valid'], device, args.out_channels)
            selection_score = current_validation_class_metrics['macro_f1']
        else:
            selection_score = result[1]

        if epoch + 1 >= args.min_checkpoint_epoch and selection_score > best_selection_score:
            best_selection_score = selection_score
            best_val = result[1]
            best_epoch = epoch
            best_train_acc = result[0]
            best_test_acc_at_best_val = result[2]
            patience = 0
            best_model_state = copy.deepcopy(model.state_dict())

            best_validation_class_metrics = (
                current_validation_class_metrics
                if current_validation_class_metrics is not None else
                validation_class_metrics(model, dataset, split_idx['valid'],
                                         device, args.out_channels))

            train_f1, val_f1, test_f1 = calculate_f1(model, dataset, split_idx, device)
            best_train_f1 = train_f1
            best_val_f1 = val_f1
            best_test_f1 = test_f1
        elif epoch + 1 >= args.min_checkpoint_epoch:
            patience += 1
            if patience >= args.patience:
                break

        if epoch % args.display_step == 0:
            if getattr(args, 'use_cl', True):
                print(f'Epoch: {epoch:02d}, '
                      f'Total Loss: {loss:.4f} (Task: {task_loss:.4f}, CL: {contrast_loss:.4f}), '
                      f'Train: {100 * result[0]:.5f}%, '
                      f'Valid: {100 * result[1]:.5f}%, '
                      f'Test: {100 * result[2]:.5f}%, '
                      f'Highest Test: {100 * best_test_acc_at_best_val:.5f}%')
                if args.cl_mode in ('shared_supcon', 'patient_prototype'):
                    head_gate = (torch.sigmoid(model.cl_head_gate).item()
                                 if getattr(model, 'cl_aux_head', False) else 0.0)
                    print(f'  CL diagnostics: supervised_contrast={model.last_supcon_loss:.4f}, '
                          f'consistency={model.last_consistency_loss:.4f}, '
                          f'input_gate={torch.sigmoid(model.cl_gate_logit).item():.4f}, '
                          f'aux_head_gate={head_gate:.4f}')
            else:
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * result[0]:.2f}%, '
                      f'Valid: {100 * result[1]:.2f}%, '
                      f'Test: {100 * result[2]:.2f}%, '
                      f'Highest Test: {100 * best_test_acc_at_best_val:.2f}%')

    logger.print_statistics(run)

    robustness_results = None
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if args.checkpoint_file:
            os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint_file)), exist_ok=True)
            torch.save({'state_dict': best_model_state, 'best_epoch': best_epoch,
                        'best_validation_accuracy': best_val,
                        'checkpoint_metric': args.checkpoint_metric,
                        'selection_score': best_selection_score,
                        'best_validation_macro_f1': best_validation_class_metrics['macro_f1'],
                        'validation_class_recall': best_validation_class_metrics['class_recall'],
                        'validation_class_support': best_validation_class_metrics['class_support']},
                       args.checkpoint_file)
        if args.embedding_file:
            model.eval()
            with torch.no_grad():
                x = dataset.graph['node_feat']
                edge = dataset.graph['edge_index']
                if model.use_adv and model.cl_mode in ('shared_supcon', 'patient_prototype'):
                    clean = (model._clean_hyper_embedding(dataset)
                             if model.cl_mode == 'patient_prototype'
                             else model.adv_module(x, return_clean=True))
                    x, _ = model._shared_input(x, clean)
                trans_hyp = model.trans_conv(x)
                trans_euc = model.manifold_out.logmap0(trans_hyp)[..., 1:]
                trans_euc = model.euc_ln(trans_euc)
                graph_euc = model.graph_conv(x, edge)
                embedding = torch.cat((trans_euc, graph_euc), dim=-1).cpu().numpy()
            os.makedirs(os.path.dirname(os.path.abspath(args.embedding_file)), exist_ok=True)
            np.savez_compressed(args.embedding_file, node_idx=np.arange(n),
                                embedding=embedding,
                                major_labels=dataset.label.squeeze(1).cpu().numpy(),
                                minor_labels=np.asarray(dataset.celltype_minor),
                                patient_ids=np.asarray(dataset.patient_ids))
        print(f"\n🚀 開始在最佳模型上進行魯棒性壓力測試與可视化 (Run {run + 1})")

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)

        if args.prediction_file:
            prediction_dir = os.path.dirname(args.prediction_file)
            if prediction_dir:
                os.makedirs(prediction_dir, exist_ok=True)
            with torch.no_grad():
                test_idx_cpu = split_idx['test'].cpu()
                test_logits = model(dataset)[split_idx['test'].to(device)]
                predictions = test_logits.argmax(dim=1).cpu()
            np.savez_compressed(
                args.prediction_file,
                test_idx=test_idx_cpu.numpy(),
                true=dataset.label.squeeze(1).cpu()[test_idx_cpu].numpy(),
                pred=predictions.numpy(),
                prob=test_logits.softmax(dim=1).cpu().numpy(),
            )

        if args.save_topology and hasattr(model, 'adv_module') and initial_topology_metrics is not None:
            model.adv_module.train()
            if args.topology_consensus > 1:
                total_evolved = None
                total_hard = None
                with torch.no_grad():
                    for _ in range(args.topology_consensus):
                        _, _, _, snap = model.adv_module(
                            dataset.graph['node_feat'].to(device))
                        h_evolved = snap['H_evolved'].float()
                        h_hard = snap['H_hard'].float()
                        total_evolved = (h_evolved if total_evolved is None
                                         else total_evolved + h_evolved)
                        total_hard = (h_hard if total_hard is None
                                      else total_hard + h_hard)
                best_snapshot = snap
                best_snapshot['H_evolved'] = (
                    total_evolved / args.topology_consensus >= 0.5
                ).to(h_evolved.dtype)
                best_snapshot['H_hard'] = (
                    total_hard / args.topology_consensus >= 0.5
                ).to(h_hard.dtype)
            else:
                with torch.no_grad():
                    _, _, _, best_snapshot = model.adv_module(
                        dataset.graph['node_feat'].to(device))
            final_topology_metrics = save_topology_snapshot(
                snapshot=best_snapshot,
                labels=all_labels,
                output_dir=topology_experiment_dir,
                dataset_name=args.dataset,
                run=run,
                epoch=best_epoch,
                tag='evolved',
                export_csv=bool(args.topology_csv),
                export_plot=bool(args.topology_plot),
                export_class_block=bool(args.topology_class_block),
                params=topology_params,
                images_subdir='figures',
            )
            if final_topology_metrics is not None:
                save_topology_comparison(
                    initial_topology_metrics,
                    final_topology_metrics,
                    output_dir=topology_experiment_dir,
                    dataset_name=args.dataset,
                    run=run,
                    best_epoch=best_epoch,
                    fixed_metrics=fixed_topology_metrics,
                    params=topology_params,
                )
            model.eval()

        final_train_f1, final_val_f1, final_test_f1 = calculate_f1(model, dataset, split_idx, device)

        robustness_results = run_robustness_test(model, dataset, split_idx, device,
                                                 seed=args.seed + run)
        all_robustness_results.append(robustness_results)
        if args.robustness_file:
            import json
            os.makedirs(os.path.dirname(os.path.abspath(args.robustness_file)), exist_ok=True)
            with open(args.robustness_file, 'w', encoding='utf-8') as stream:
                json.dump(robustness_results, stream, indent=2)

    if args.output_attention:
        attentions = model.get_attentions(dataset.graph['node_feat'].to(device))
        np.save(f'results/att/{args.dataset}_{args.method}_{args.hidden_channels}_attentions.npy',
                attentions.detach().cpu().numpy())

    result_saver.save_run_result(
        run_idx=run,
        train_acc=best_train_acc * 100,
        val_acc=best_val * 100,
        highest_test_acc=best_test_acc_at_best_val * 100,
        final_train_acc=result[0] * 100,
        final_test_acc=result[2] * 100,
        train_f1=best_train_f1,
        val_f1=best_val_f1,
        test_f1=best_test_f1,
        final_train_f1=final_train_f1,
        final_test_f1=final_test_f1,
        robustness_edge=robustness_results['edge'] if robustness_results else None,
        robustness_feat=robustness_results['feat'] if robustness_results else None,
        method=args.method,
        use_cl=getattr(args, 'use_cl', False),
        use_adaptive_curvature=getattr(args, 'use_adaptive_curvature', False),
        cl_space=getattr(args, 'cl_space', 'hyp'),
        graph_build_method=getattr(args, 'graph_build_method', 'knn'),
        knn_metric=getattr(args, 'knn_metric', 'hyp'),
        drop_strategy=getattr(args, 'drop_strategy', 'critical')
    )

    del model, optimizer

if args.runs > 1:
    results = logger.print_statistics()
    print('====================')
    print(results)
    print('====================')

    f1_summary = result_saver.calculate_and_save_averages(args.runs)
    result_saver.save_summary(results, all_robustness_results, args=args, f1_summary=f1_summary)

out_folder = 'results'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

if args.save_result:
    mkdirs(f'results/{args.dataset}')
    csvfilename = f'results/{args.dataset}/{args.dataset}_{args.method}_{args.hidden_channels}.csv'
    logger.save(vars(args), results, csvfilename)
