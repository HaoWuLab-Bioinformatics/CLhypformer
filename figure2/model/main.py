import argparse
import copy
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
from sklearn.metrics import f1_score

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
    torch.use_deterministic_algorithms = True


def pretrain_contrastive_module(model, dataset, args, device):
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

        z1_hyp, z2_hyp, contrast_loss, _ = model.adv_module(x)

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


def run_robustness_test(model, dataset, split_idx, device):
    """
    在測試集上執行魯棒性壓力測試（無數據洩露版本）
    """
    model.eval()
    drop_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
    test_idx = split_idx['test'].to(device)
    labels = dataset.label.squeeze(1)[test_idx]

    orig_edge = dataset.graph['edge_index']
    orig_edge_tensor = orig_edge[0] if isinstance(orig_edge, tuple) else orig_edge
    orig_feat_tensor = dataset.graph['node_feat'].clone()

    robustness_results = {'edge': {}, 'feat': {}}

    print("\n" + "=" * 55)
    print("🛡️ 結構魯棒性測試 (Structural Noise - Edge Deletion)")
    print("=" * 55)

    with torch.no_grad():
        for dr in drop_rates:
            new_edge = drop_edges(orig_edge_tensor, dr)
            if isinstance(orig_edge, tuple):
                dataset.graph['edge_index'] = (new_edge,)
            else:
                dataset.graph['edge_index'] = new_edge

            out = model(dataset)
            if isinstance(out, tuple): out = out[0]

            pred = out[test_idx].max(1)[1]
            acc = pred.eq(labels).sum().item() / len(labels)
            robustness_results['edge'][dr] = acc * 100
            print(f"🔹 Drop Rate {dr * 100:2.0f}% -> Test Acc: {acc * 100:.2f}%")

            dataset.graph['edge_index'] = orig_edge

    dataset.graph['edge_index'] = orig_edge

    print("\n" + "=" * 55)
    print("🛡️ 特徵魯棒性測試 (Feature Noise - Node Masking)")
    print("=" * 55)

    with torch.no_grad():
        for dr in drop_rates:
            noisy_feat = orig_feat_tensor.clone()
            test_mask = (torch.rand_like(noisy_feat[test_idx]) > dr).to(noisy_feat.dtype)
            noisy_feat[test_idx] = noisy_feat[test_idx] * test_mask

            dataset.graph['node_feat'] = noisy_feat.to(device)

            out = model(dataset)
            if isinstance(out, tuple): out = out[0]

            pred = out[test_idx].max(1)[1]
            acc = pred.eq(labels).sum().item() / len(labels)
            robustness_results['feat'][dr] = acc * 100
            print(f"🔹 Drop Rate {dr * 100:2.0f}% -> Test Acc: {acc * 100:.2f}%")

            dataset.graph['node_feat'] = orig_feat_tensor.to(device)

    dataset.graph['node_feat'] = orig_feat_tensor.to(device)
    print("=" * 55 + "\n")

    return robustness_results


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
parser.add_argument('--cl_knn', type=int, default=5, help='构建KNN对比图的邻居数K')
parser.add_argument('--cl_drop_rate', type=float, default=0.5, help='增强视图随机丢弃边的概率')
parser.add_argument('--cl_tau', type=float, default=0.5, help='双曲对比损失的温度系数')
parser.add_argument('--do_pretrain', action='store_true', default=True, help='是否开启对比学习预训练')
parser.add_argument('--pretrain_epochs', type=int, default=50, help='对比学习预训练的轮数')
parser.add_argument('--use_adaptive_curvature', action='store_true', default=False,
                    help='是否开启自适应曲率 (Adaptive Curvature)')
parser.add_argument('--knn_metric', type=str, default='hyp', choices=['hyp', 'euc'],
                    help='构建KNN图的度量空间 (hyp或euc)')
parser.add_argument('--cl_space', type=str, default='hyp', choices=['hyp', 'euc'], help='对比学习所处空间 (hyp或euc)')
parser.add_argument('--graph_build_method', type=str, default='knn', choices=['knn', 'random_walk'],
                    help='图构建方式: knn或距离阈值eps或随机游走random_walk')
parser.add_argument('--drop_strategy', type=str, default='topk', choices=['critical', 'random', 'topk'],
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
args = parser.parse_args()
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

if args.rand_split:
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
        model = pretrain_contrastive_module(model, dataset, args, device)

    if getattr(args, 'use_cl', True):
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

        labels = dataset.label.squeeze(1)[train_idx]

        snapshot = None
        if getattr(args, 'use_cl', True):
            out_orig, out1, out2, contrast_loss = model(dataset)
            if args.save_topology and hasattr(model, 'adv_module'):
                snapshot = getattr(model.adv_module, 'last_snapshot', None)
            loss_orig = F.nll_loss(F.log_softmax(out_orig[train_idx], dim=-1), labels)

            if out2 is None:
                loss1 = F.nll_loss(F.log_softmax(out1[train_idx], dim=-1), labels)
                task_loss = loss_orig + 0.1 * loss1
            else:
                loss1 = F.nll_loss(F.log_softmax(out1[train_idx], dim=-1), labels)
                loss2 = F.nll_loss(F.log_softmax(out2[train_idx], dim=-1), labels)
                task_loss = loss_orig + 0.1 * (loss1 + loss2)

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
                and getattr(args, 'use_cl', True)
                and hasattr(model, 'adv_module')):
            topology_homophily_loss = model.adv_module.embedding_homophily_loss(
                dataset.graph['node_feat'], dataset.label.squeeze(1), train_idx)
            loss = loss + args.topology_label_weight * topology_homophily_loss

        loss.backward()
        optimizer.step()

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            best_epoch = epoch
            best_train_acc = result[0]
            best_test_acc_at_best_val = result[2]
            patience = 0
            best_model_state = copy.deepcopy(model.state_dict())

            train_f1, val_f1, test_f1 = calculate_f1(model, dataset, split_idx, device)
            best_train_f1 = train_f1
            best_val_f1 = val_f1
            best_test_f1 = test_f1
        else:
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
        print(f"\n🚀 開始在最佳模型上進行魯棒性壓力測試與可视化 (Run {run + 1})")

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)

        if args.dataset == 'cora' and run == 0 and hasattr(model, 'decode_trans'):
            features = []
            def capture_decoder_input(module, inputs):
                features.append(inputs[0].detach().cpu().numpy())
            hooks = [model.decode_trans.register_forward_pre_hook(capture_decoder_input)]
            if model.use_graph:
                hooks.append(model.decode_graph.register_forward_pre_hook(capture_decoder_input))
            try:
                model.eval()
                with torch.no_grad():
                    model(dataset)
            finally:
                for hook in hooks:
                    hook.remove()
            np.savez_compressed('cora_embedding.npz', points=np.concatenate(features, axis=1),
                                labels=dataset.label.detach().cpu().numpy().reshape(-1))

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

        robustness_results = run_robustness_test(model, dataset, split_idx, device)
        all_robustness_results.append(robustness_results)

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
