#!/usr/bin/env python3
"""
自定义训练脚本 - 支持灵活的参数配置与 t-SNE 可视化
用于DHG-Bench超图神经网络训练
"""
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import csv

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime
from lib_utils.exp_agent import ExpAgent
from lib_models.HNN.preprocessing import algo_preprocessing
from lib_dataset.data_base import HyperDataset
from lib_dataset.preprocessing import data_processing
from parameter_parser import method_config, set_task_args


# ================= 辅助函数：从 ExpAgent 提取 Model =================
def extract_model(agent):
    """
    专为 DHG-Bench 的 ExpAgent/Trainer 结构设计的模型提取器
    支持检索下划线私有属性（如 _model, _net 等）
    """
    candidates = []
    if hasattr(agent, 'trainer') and agent.trainer is not None:
        candidates.append(agent.trainer)
    if hasattr(agent, 'evaluator') and agent.evaluator is not None:
        candidates.append(agent.evaluator)
    candidates.append(agent)

    for obj in candidates:
        if isinstance(obj, torch.nn.Module):
            return obj

        all_attrs = set()
        if hasattr(obj, '__dict__'):
            all_attrs.update(obj.__dict__.keys())
        all_attrs.update(dir(obj))

        for attr in all_attrs:
            if attr.startswith('__') and attr.endswith('__'):
                continue
            try:
                val = getattr(obj, attr)
                if isinstance(val, torch.nn.Module):
                    return val
            except Exception:
                continue

    for obj in candidates:
        all_attrs = set()
        if hasattr(obj, '__dict__'):
            all_attrs.update(obj.__dict__.keys())
        all_attrs.update(dir(obj))

        for attr in all_attrs:
            if attr.startswith('__') and attr.endswith('__'):
                continue
            try:
                val = getattr(obj, attr)
                if val is not None and not isinstance(
                        val, (int, float, str, bool, list, dict, tuple, set, torch.Tensor)
                ):
                    val_attrs = set()
                    if hasattr(val, '__dict__'):
                        val_attrs.update(val.__dict__.keys())
                    val_attrs.update(dir(val))
                    for sub_attr in val_attrs:
                        if sub_attr.startswith('__') and sub_attr.endswith('__'):
                            continue
                        sub_val = getattr(val, sub_attr)
                        if isinstance(sub_val, torch.nn.Module):
                            return sub_val
            except Exception:
                continue

    return None


# ================= 可视化核心函数 =================
def visualize_embeddings(embeddings, labels, save_path, cl_space='hyp', manifold=None):
    """
    对模型学习到的节点嵌入进行 t-SNE 可视化并保存
    支持双曲几何到欧式切空间的对数映射，并自动处理标签与嵌入维度不一致的问题（已移除图例）
    """
    embeddings = embeddings.detach()

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if embeddings.dim() == 3:
        embeddings = embeddings.squeeze(0)

    if cl_space == 'hyp':
        if manifold is None:
            print("⚠️ 警告: cl_space='hyp' 但未传入 manifold，默认使用原始输出。")
            embeddings_euc = embeddings
        else:
            embeddings_euc = manifold.logmap0(embeddings)[..., 1:]
    else:
        embeddings_euc = embeddings

    X = embeddings_euc.cpu().numpy()

    # ================= 尺寸校验与安全处理 =================
    if len(labels) != X.shape[0]:
        print(f"⚠️ 警告: 嵌入样本数 ({X.shape[0]}) 与标签数 ({len(labels)}) 不一致！")
        print(f"   -> 自动调整标签数组以匹配嵌入维度，避免绘图崩溃。")
        if len(labels) > X.shape[0]:
            labels = labels[:X.shape[0]]
        else:
            padded_labels = np.zeros(X.shape[0], dtype=labels.dtype if hasattr(labels, 'dtype') else int)
            padded_labels[:len(labels)] = labels
            labels = padded_labels
    # =====================================================

    print(f"🎬 开始运行 t-SNE 降维 (样本数: {X.shape[0]}, 特征维度: {X.shape[1]})...")

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(8, 7), dpi=300)

    # 绘制散点图（移除了图例相关代码）
    scatter = plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=labels,
        cmap='tab10',
        s=15,
        alpha=0.8,
        edgecolors='none'
    )

    plt.xticks([])
    plt.yticks([])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"🎉 可视化图片已成功保存至: {save_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DHG-Bench 自定义训练脚本')

    parser.add_argument('--dataset', type=str, default='twitch_ES', help='数据集名称')
    parser.add_argument('--model', type=str, default='HGNN', help='模型名称')
    parser.add_argument('--task', type=str, default='node_cls', choices=['node_cls', 'edge_pred', 'hg_cls'],
                        help='任务类型')

    parser.add_argument('--train_ratio', type=float, default=0.5, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.25, help='验证集比例')
    parser.add_argument('--use_bench_prop', action='store_true', default=False, help='使用基准划分比例')

    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='权重衰减')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout比率')

    parser.add_argument('--embedding_hidden', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--All_num_layers', type=int, default=2, help='网络层数')
    parser.add_argument('--MLP_hidden', type=int, default=128, help='MLP隐藏层维度')

    parser.add_argument('--device', type=str, default='cuda:0', help='计算设备')
    parser.add_argument('--num_runs', type=int, default=5, help='训练次数')

    parser.add_argument('--early_stop', action='store_true', default=True, help='启用早停')
    parser.add_argument('--display_step', type=int, default=10, help='显示间隔')
    parser.add_argument('--is_default', action='store_true', default=False, help='使用默认配置')

    parser.add_argument('--robustness_test', action='store_true', default=False, help='启用鲁棒性测试')
    parser.add_argument('--perturb_type', type=str, default='feature_noise',
                        choices=['feature_noise', 'label_noise', 'edge_drop'], help='扰动类型')
    parser.add_argument('--perturb_levels', type=float, nargs='+', default=[0.1, 0.2, 0.3], help='扰动强度列表')

    return parser.parse_args()


def build_args_from_config(config):
    """从配置字典构建完整的args对象"""
    from argparse import Namespace
    args = Namespace()

    args.dname = config.get('dataset', 'cora')
    args.method = config.get('model', 'HGNN')
    args.task_type = config.get('task', 'node_cls')

    args.use_bench_prop = config.get('use_bench_prop', False)
    if not args.use_bench_prop:
        args.train_prop = config.get('train_ratio', 0.5)
        args.valid_prop = config.get('val_ratio', 0.25)
    else:
        if args.task_type == 'node_cls':
            args.train_prop, args.valid_prop = 0.5, 0.25
        elif args.task_type == 'hg_cls':
            args.train_prop, args.valid_prop = 0.8, 0.1
        else:
            args.train_prop, args.valid_prop = 0.6, 0.2

    args.epochs = config.get('epochs', 100)
    args.lr = config.get('lr', 0.001)
    args.wd = config.get('weight_decay', 0.0)
    args.dropout = config.get('dropout', 0.5)
    args.hg_batch_size = config.get('batch_size', 256)

    args.embedding_mode = True
    args.embedding_hidden = config.get('embedding_hidden', 128)
    args.All_num_layers = config.get('All_num_layers', 2)
    args.MLP_hidden = config.get('MLP_hidden', 128)
    args.HCHA_symdegnorm = config.get('HCHA_symdegnorm', False)
    args.normtype = 'all_one'
    args.add_self_loop = False
    args.exclude_self = False

    args.device = config.get('device', 'cuda:0')
    args.num_runs = config.get('num_runs', 5)

    args.early_stop = config.get('early_stop', True)
    args.clip_grad = False
    args.clip_thresh = 5.0

    args.display_step = config.get('display_step', 10)
    args.eval_verbose = True
    args.mem_verbose = True
    args.mem_display_step = 100

    args.use_processed = True
    args.is_default = config.get('is_default', False)

    args.edge_split_mode = 'ind'
    args.edge_save_dir = './lib_edge_splits/'
    args.edge_batch_size = 512
    args.e_embed_hidden = 64
    args.e_embed_layer = 2
    args.e_embed_dropout = 0.2
    args.e_embed_norm = 'ln'
    args.aggr_mode = 'max'
    args.ns_method = 'mixed'
    args.edge_aggr = 'group'

    args.pooling = 'mean'
    args.g_embed_hidden = 128
    args.g_embed_layer = 2
    args.g_embed_dropout = 0.2
    args.g_embed_norm = 'ln'
    args.use_weighted_loss = False

    args.is_perturbed = config.get('is_perturbed', False)
    args.pert_mode = config.get('pert_mode', 'spar_label')
    args.pert_p = config.get('pert_p', 0.0)
    args.is_poison = True
    args.feature_noise = config.get('feature_noise', '0.6')

    return args


def save_results_to_dataset_folder(args, acc_mean, acc_std, all_accs):
    """将结果保存到以数据集命名的文件夹中"""
    results_base = Path(__file__).parent / 'results'
    dataset_folder = results_base / args.dname
    dataset_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = dataset_folder / f'{args.method}_acc{acc_mean:.4f}_std{acc_std:.4f}_{timestamp}.csv'

    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Dataset', 'Task', 'ACC_Mean', 'ACC_Std', 'Num_Runs', 'Timestamp'])
        writer.writerow([
            args.method,
            args.dname,
            args.task_type,
            f'{acc_mean:.4f}',
            f'{acc_std:.4f}',
            len(all_accs),
            timestamp
        ])

    print(f'\n📁 结果已保存到: {csv_file}')
    print(f'   数据集: {args.dname}')
    print(f'   模型: {args.method}')
    print(f'   平均ACC: {acc_mean:.4f} ± {acc_std:.4f}')

    return csv_file


def train_single_run(config, run_idx=1):
    """单次训练运行"""
    print(f"\n{'=' * 70}")
    print(f"🚀 第 {run_idx} 次训练")
    print(f"{'=' * 70}")

    args = build_args_from_config(config)
    args = method_config(args)
    args = set_task_args(args)

    data = HyperDataset(args)

    if args.task_type == 'hg_cls':
        data = data.multi_hypergraphs
    else:
        data = data_processing(args, data)
        data._initialization_()
        data = algo_preprocessing(data, args)

    captured_modules = []
    original_init = torch.nn.Module.__init__

    def patched_init(self, *args_i, **kwargs_i):
        original_init(self, *args_i, **kwargs_i)
        captured_modules.append(self)

    torch.nn.Module.__init__ = patched_init

    try:
        agent = ExpAgent(args)
        agent.running(args.task_type, data)

        if hasattr(agent, 'test_dict') and 'acc' in agent.test_dict:
            test_acc = agent.test_dict['acc'][0]
        else:
            print("警告: 无法获取测试准确率")
            test_acc = 0.0
    finally:
        torch.nn.Module.__init__ = original_init

    print("\n🎨 正在为当前配置生成 t-SNE 可视化图片...")
    try:
        valid_candidates = []
        for m in captured_modules:
            try:
                params_count = sum(p.numel() for p in m.parameters())
                if params_count > 0:
                    valid_candidates.append((m, params_count))
            except Exception:
                pass

        valid_candidates.sort(key=lambda x: x[1], reverse=True)

        target_model = None
        for candidate, params_cnt in valid_candidates:
            try:
                candidate.eval()
                with torch.no_grad():
                    _ = candidate(data)
                target_model = candidate
                print(f"🎯 成功锁定主模型: {target_model.__class__.__name__} (参数量: {params_cnt})")
                break
            except Exception:
                continue

        if target_model is None:
            print("⚠️ 未能找到适配当前 data 的顶层框架模型。")
        else:
            data = data.to(args.device)
            target_model.eval()
            with torch.no_grad():
                model_out = target_model(data)
                if isinstance(model_out, tuple):
                    logits = model_out[0]
                    embeddings = logits
                else:
                    embeddings = model_out

            cl_space = getattr(args, 'cl_space', 'hyp') if hasattr(target_model,
                                                                   'manifold') and target_model.manifold is not None else 'euc'
            graph_build = getattr(args, 'graph_build_method', 'knn')
            knn_metric = getattr(args, 'knn_metric', 'euc')

            fig_name = f"{args.dname}_{cl_space}_{graph_build}_{knn_metric}_run{run_idx}.png"
            save_fig_path = os.path.join("./figs", fig_name)

            if hasattr(data, 'label'):
                labels = data.label.squeeze() if isinstance(data.label,
                                                            torch.Tensor) and data.label.dim() > 1 else data.label
            elif hasattr(data, 'y'):
                labels = data.y.squeeze() if isinstance(data.y, torch.Tensor) and data.y.dim() > 1 else data.y
            else:
                labels = np.zeros(embeddings.shape[0])

            visualize_embeddings(
                embeddings=embeddings,
                labels=labels,
                save_path=save_fig_path,
                cl_space=cl_space,
                manifold=getattr(target_model, 'manifold', None) if cl_space == 'hyp' else None
            )
    except Exception as e:
        print(f"⚠️ 可视化过程发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        captured_modules.clear()

    return test_acc


def train_multiple_runs(config):
    """多次独立训练并计算平均ACC"""
    num_runs = config.get('num_runs', 5)

    print("\n" + "=" * 70)
    print(f"开始 {num_runs} 次训练")
    print("=" * 70)

    all_accs = []
    for run_idx in range(1, num_runs + 1):
        try:
            test_acc = train_single_run(config, run_idx)
            all_accs.append(test_acc)
            print(f"\n>>> 第 {run_idx}/{num_runs} 次训练完成 - ACC: {test_acc:.4f}")
        except Exception as e:
            print(f"\n!!! 第 {run_idx} 次训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_accs) == 0:
        print("\n❌ 所有训练都失败了！")
        return 0.0, 0.0, []

    acc_mean = np.mean(all_accs)
    acc_std = np.std(all_accs)

    print("\n" + "=" * 70)
    print("📊 最终统计结果")
    print(f"  平均 ACC: {acc_mean:.4f}")
    print(f"  标准差:   {acc_std:.4f}")
    print(f"  训练次数: {len(all_accs)}")
    print("=" * 70)

    return acc_mean, acc_std, all_accs


def robustness_test(config):
    perturb_type = config.get('perturb_type', 'feature_noise')
    perturb_levels = config.get('perturb_levels', [0.1, 0.2, 0.3])

    print("\n" + "=" * 70)
    print("鲁棒性测试")
    print(f"扰动类型: {perturb_type}")
    print(f"扰动水平: {perturb_levels}")
    print("=" * 70)

    base_config = config.copy()
    base_config['is_perturbed'] = False
    base_acc_mean, base_acc_std, base_all_accs = train_multiple_runs(base_config)

    results_dict = {}
    for level in perturb_levels:
        robust_config = config.copy()
        robust_config['is_perturbed'] = True
        robust_config['pert_p'] = level

        if perturb_type == 'feature_noise':
            robust_config['pert_mode'] = 'noise_feat'
            robust_config['feature_noise'] = str(level)
        elif perturb_type == 'label_noise':
            robust_config['pert_mode'] = 'flip_label'
        elif perturb_type == 'edge_drop':
            robust_config['pert_mode'] = 'drop_incidence'

        acc_mean, acc_std, all_accs = train_multiple_runs(robust_config)
        results_dict[level] = {'acc_mean': acc_mean, 'acc_std': acc_std, 'all_accs': all_accs}

    return results_dict


def train_with_config(config):
    print("=" * 70)
    print("DHG-Bench 自定义训练脚本")
    print("=" * 70)

    if config.get('robustness_test', False):
        robustness_test(config)
    else:
        acc_mean, acc_std, all_accs = train_multiple_runs(config)
        if len(all_accs) > 0:
            args = build_args_from_config(config)
            save_results_to_dataset_folder(args, acc_mean, acc_std, all_accs)

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)


def quick_train():
    config = {
        'dataset': 'cora',
        'model': 'HGNN',
        'task': 'node_cls',
        'train_ratio': 0.5,
        'val_ratio': 0.25,
        'use_bench_prop': False,
        'epochs': 100,
        'lr': 0.001,
        'weight_decay': 0.0,
        'dropout': 0.5,
        'batch_size': 256,
        'embedding_hidden': 128,
        'All_num_layers': 2,
        'MLP_hidden': 128,
        'device': 'cuda:0',
        'num_runs': 1,
        'early_stop': True,
        'display_step': 10,
        'robustness_test': True,
        'perturb_type': 'edge_drop',
        'perturb_levels': [0, 0.2, 0.4, 0.6, 0.8],
    }
    train_with_config(config)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = parse_args()
        config = {
            'dataset': args.dataset,
            'model': args.model,
            'task': args.task,
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'use_bench_prop': args.use_bench_prop,
            'epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'dropout': args.dropout,
            'embedding_hidden': args.embedding_hidden,
            'All_num_layers': args.All_num_layers,
            'MLP_hidden': args.MLP_hidden,
            'device': args.device,
            'num_runs': args.num_runs,
            'early_stop': args.early_stop,
            'display_step': args.display_step,
            'is_default': args.is_default,
            'robustness_test': args.robustness_test,
            'perturb_type': args.perturb_type,
            'perturb_levels': args.perturb_levels,
        }
        train_with_config(config)
    else:
        quick_train()