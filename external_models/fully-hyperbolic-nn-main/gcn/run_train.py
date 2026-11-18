"""
全双曲神经网络训练脚本
支持单次运行或多次运行取平均值，并自动在训练后进行 t-SNE 可视化及鲁棒性测试
"""

import sys
import os
from pathlib import Path
import numpy as np
import json
import argparse
import copy
import torch
import scipy.sparse as sp

# 添加 gcn 目录到 Python 路径
gcn_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(gcn_dir))

from train import train
from utils.data_utils import edge_mask_adj
from argparse import Namespace
import manifolds


# ==================== 补丁：确保 Lorentz 流形拥有 logmap0 方法 ====================
def patch_lorentz_logmap0():
    """为 Lorentz 流形补全 logmap0 方法，防止 visualize.py 调用时抛出 AttributeError"""
    LorentzClass = getattr(manifolds, 'Lorentz', None)
    if LorentzClass is not None and not hasattr(LorentzClass, 'logmap0'):
        def logmap0(self, x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)

            k = getattr(self, 'k', 1.0)
            if torch.is_tensor(k):
                k_val = k.item()
            else:
                k_val = float(k)
            sqrt_k = np.sqrt(max(k_val, 1e-7))

            x0 = x[..., 0:1] / sqrt_k
            x0 = torch.clamp(x0, min=1.0 + 1e-7)

            dist = torch.acosh(x0)
            sinh_dist = torch.sinh(dist)
            scale = torch.where(sinh_dist > 1e-7, dist / sinh_dist, torch.ones_like(dist))

            spatial = x[..., 1:] * scale
            zeros = torch.zeros_like(x[..., 0:1])
            return torch.cat([zeros, spatial], dim=-1)

        LorentzClass.logmap0 = logmap0

patch_lorentz_logmap0()
# ==============================================================================

# ==================== 核心修复：图结构重新归一化 ====================
def recompute_adj_norm(adj):
    """
    为鲁棒性测试重新计算对称归一化的邻接矩阵 Tensor (带自环)
    直接写在主脚本中，避免 utils 跨文件导入失败导致崩溃
    """
    # 如果传来的是 Tensor，先转换为 scipy 稀疏矩阵
    if torch.is_tensor(adj):
        if adj.is_sparse:
            adj = adj.to_dense()
        adj = sp.csr_matrix(adj.cpu().numpy())

    # 1. 确保带有自环 (Self-loops)
    adj = adj + sp.eye(adj.shape[0])

    # 2. 对称归一化: D^{-1/2} * A * D^{-1/2}
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_norm = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    # 3. 转换为 PyTorch 的 Sparse Tensor
    indices = torch.from_numpy(np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64))
    values = torch.from_numpy(adj_norm.data.astype(np.float32))
    shape = torch.Size(adj_norm.shape)

    return torch.sparse.FloatTensor(indices, values, shape)
# ==============================================================================


def get_base_config():
    """获取基础配置"""
    return {
        'dataset': 'cora',
        'task': 'nc',
        'model': 'HyboNet',
        'manifold': 'Lorentz',
        'cuda': 0,
        'lr': 0.001,
        'epochs': 4000,
        'patience': 800,
        'min_epochs': 150,
        'dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'weight_decay': 1e-3,
        'optimizer': 'radam',
        'bias': 1,
        'act': 'relu',
        'grad_clip': 1.0,
        'seed': None,
        'log_freq': 5,
        'eval_freq': 1,
        'lr_reduce_freq': 800,
        'gamma': 0.9,
        'momentum': 0.999,
        'double_precision': False,
        'save': 1,
        'save_dir': None,
        'c': 1.0,
        'r': 2.0,
        't': 1.0,
        'margin': 1.0,
        'n_heads': 4,
        'alpha': 0.2,
        'use_att': 0,
        'local_agg': 0,
        'val_prop': 0.05,
        'test_prop': 0.1,
        'use_feats': 1,
        'normalize_feats': 1,
        'normalize_adj': 1,
        'split_seed': None,
        'pretrained_embeddings': None,
        'pos_weight': 1,
        'visualize': 0,
        'edge_masking_rates': [0,0.2, 0.4, 0.6, 0.8],
        'masking_seed': 1234
    }


def do_visualization(save_dir, manifold_name='Lorentz'):
    """从 save_dir 加载文件并生成 t-SNE 图像"""
    if not save_dir or not os.path.exists(save_dir):
        print(f"\n[可视化提示] 目录 {save_dir} 不存在，跳过可视化。")
        return False

    emb_file = os.path.join(save_dir, 'embeddings.npy')
    label_file = os.path.join(save_dir, 'labels.npy')

    if not os.path.exists(emb_file) or not os.path.exists(label_file):
        print(f'\n[警告] 在 {save_dir} 未找到 embeddings.npy 或 labels.npy，跳过可视化。')
        return False

    try:
        from visualize import visualize_from_npy
        viz_file = os.path.join(save_dir, 'tsne_visualization.png')
        print(f'\n[可视化阶段] 正在读取特征数据并生成 t-SNE 图像...')
        visualize_from_npy(emb_file, label_file, viz_file, manifold_name=manifold_name)
        print(f'√ 可视化图像已成功保存至: {viz_file}')
        return True
    except Exception as e:
        print(f'\n[可视化出错] 生成可视化图表时抛出异常: {e}')
        return False


def run_single_experiment(run_id, base_config, should_visualize=True):
    """运行单次实验"""
    print(f"\n{'='*60}")
    print(f"开始第 {run_id} 次运行")
    print(f"{'='*60}\n")

    config = base_config.copy()
    seed = 1234 + run_id
    config['seed'] = seed
    config['split_seed'] = seed
    config['save'] = 1

    if config.get('save_dir') is None:
        import datetime
        from utils.train_utils import get_dir_name
        date_str = datetime.datetime.now().strftime('%Y_%m_%d')
        log_dir = os.environ.get('LOG_DIR', str(gcn_dir / 'logs'))
        models_dir = os.path.join(log_dir, config['task'], date_str)
        config['save_dir'] = get_dir_name(models_dir)

    args = Namespace(**config)

    try:
        res = train(args)
        val_metrics, test_metrics = res if isinstance(res, tuple) and len(res) == 2 else ({}, res if isinstance(res, dict) else {})

        if test_metrics:
            print(f"\n第 {run_id} 次运行测试结果:")
            for k, v in sorted(test_metrics.items()):
                print(f"  {k}: {v:.8f}")

        if should_visualize:
            do_visualization(args.save_dir, config.get('manifold', 'Lorentz'))

        return {'run_id': run_id, 'seed': seed, 'val_metrics': val_metrics, 'test_metrics': test_metrics, 'success': True, 'save_dir': args.save_dir}
    except Exception as e:
        print(f"\n第 {run_id} 次运行失败: {e}")
        return {'run_id': run_id, 'seed': seed, 'success': False, 'error': str(e)}


def run_robustness_test(base_config, masking_rates, seed):
    """执行鲁棒性 (Edge Masking) 实验"""
    import datetime
    from argparse import Namespace
    from utils.data_utils import load_data

    print("\n" + "=" * 80)
    print("🚀 正在启动鲁棒性实验 (Edge Masking Robustness Test)")
    print("=" * 80)
    print(f"边遮掩率列表 (Masking rates): {masking_rates}")
    print(f"遮掩随机种子 (Seed): {seed}")
    print("=" * 80 + "\n")

    config = base_config.copy()
    config["seed"] = seed
    config["split_seed"] = seed
    config["save"] = 1
    args_template = Namespace(**config)

    datapath = str(Path(__file__).parent / "data" / "raw" / config["dataset"])
    original_data = load_data(args_template, datapath)
    all_results = {}

    for rate in masking_rates:
        print("\n" + "-" * 70)
        print(f"  当前 Edge masking rate: {rate}")
        print("-" * 70)

        data_copy = copy.deepcopy(original_data)

        # 1. 对邻接矩阵进行 Masking
        data_copy['adj_train'] = edge_mask_adj(data_copy['adj_train'], rate, seed=seed + 100)

        # 2. 丢弃旧的归一化矩阵，并重算新的 Tensor 传入模型
        data_copy.pop('adj_train_norm', None)
        data_copy['adj_train_norm'] = recompute_adj_norm(data_copy['adj_train'])

        run_config = config.copy()
        if run_config.get("save_dir") is None:
            date_str = datetime.datetime.now().strftime("%Y_%m_%d")
            log_dir = os.environ.get("LOG_DIR", str(gcn_dir / "logs"))
            run_config["save_dir"] = os.path.join(
                log_dir, "robustness", run_config["task"], date_str,
                f"mask{int(rate*100)}"
            )
            os.makedirs(run_config["save_dir"], exist_ok=True)

        args = Namespace(**run_config)

        try:
            val_metrics, test_metrics = train(args, data=data_copy)
            all_results[rate] = {"val": val_metrics, "test": test_metrics, "success": True}
            print(f"  >> Rate {rate:.1f} 测试完成，关键指标:")
            for k, v in sorted(test_metrics.items()):
                print(f"       {k}: {v:.8f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[rate] = {"val": {}, "test": {}, "success": False, "error": str(e)}
            print(f"  >> Rate {rate:.1f}: 运行失败 - {e}")

    # ============ 打印汇总表 ============
    print("\n" + "=" * 80)
    print("鲁棒性测试汇总 (Robustness Test Summary)")
    print("=" * 80)
    dataset = base_config.get("dataset", "unknown")
    model = base_config.get("model", "unknown")

    all_metric_names = set()
    for r, res in all_results.items():
        if res['success']:
            all_metric_names.update(res["test"].keys())
    all_metric_names = sorted(all_metric_names)

    header = f"{'Rate':>6s}"
    for m in all_metric_names:
        header += f"  {m:>10s}"
    print(header)
    print("-" * len(header))

    for r in masking_rates:
        if r in all_results:
            res = all_results[r]
            if res['success']:
                row = f"{r:>6.1f}"
                for m in all_metric_names:
                    val = res["test"].get(m, float("nan"))
                    row += f"  {val:>10.8f}"
                print(row)
            else:
                print(f"{r:>6.1f}  {'FAILED':>10s}")
    print('-' * len(header))

    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    summary_file = gcn_dir / "results" / f"{dataset}_{model}_robustness_{date_str}.json"
    os.makedirs(summary_file.parent, exist_ok=True)

    summary_data = {
        "dataset": dataset,
        "model": model,
        "manifold": base_config.get("manifold", "unknown"),
        "masking_rates": list(masking_rates),
        "masking_seed": seed,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {str(r): {"val_metrics": res["val"], "test_metrics": res["test"], "success": res["success"]} for r, res in all_results.items() if res["success"]}
    }

    with open(summary_file, "w", encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 鲁棒性实验结果已保存至: {summary_file}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='全双曲神经网络训练脚本')
    # === 关键修复：添加 run-mode 参数，明确告诉脚本该跑什么 ===
    parser.add_argument('--run-mode', type=str, default='robustness', choices=['normal', 'robustness', 'viz'],
                        help='运行模式: normal(普通训练), robustness(鲁棒性测试), viz(仅可视化)')
    parser.add_argument('--num-runs', type=int, default=1, help='运行次数 (默认: 1)')
    parser.add_argument('--start-run', type=int, default=0, help='起始运行编号 (默认: 0)')
    parser.add_argument('--visualize', type=int, default=1, help='是否在普通训练后生成可视化 (默认: 1)')
    parser.add_argument('--viz-only-path', type=str, default=None, help='提供 embeddings.npy 的路径以进行独立可视化')
    parser.add_argument('--edge-masking-rates', type=float, nargs='+', default=None, help='Edge masking rates (例如 0.2 0.4 0.6 0.8)')
    args = parser.parse_args()

    # 路径初始化
    datapath = str((gcn_dir / 'data' / 'raw').resolve())
    os.environ['DATAPATH'] = datapath
    os.environ['LOG_DIR'] = str((gcn_dir / 'logs').resolve())
    os.environ['HGCN_HOME'] = str(gcn_dir.parent.resolve())
    base_config = get_base_config()

    # === 分支 1：独立可视化模式 ===
    if args.run_mode == 'viz' or args.viz_only_path:
        viz_path = args.viz_only_path
        if not viz_path or not os.path.exists(viz_path):
            print("错误：请通过 --viz-only-path 提供有效的 embeddings.npy 路径。")
            sys.exit(1)
        from visualize import visualize_from_npy
        base_dir = os.path.dirname(viz_path)
        visualize_from_npy(viz_path, os.path.join(base_dir, "labels.npy"), os.path.join(base_dir, 'tsne_visualization.png'), manifold_name='Lorentz')
        sys.exit(0)

    # === 分支 2：鲁棒性测试模式 ===
    if args.run_mode == 'robustness':
        # 优先读取命令行传递的 rates，否则 fallback 到基础配置里的 [0.2, 0.4, 0.6, 0.8]
        mask_rates = args.edge_masking_rates if args.edge_masking_rates else base_config.get('edge_masking_rates')
        run_robustness_test(base_config, mask_rates, seed=1234)
        return  # 鲁棒性跑完就直接结束

    # === 分支 3：标准单次/多次训练模式 ===
    print(f"\n======================================")
    print(f"进入标准训练模式 (数据集: {base_config['dataset']})")
    print(f"======================================\n")
    should_visualize = bool(args.visualize or base_config.get('visualize', 0))
    if args.num_runs == 1:
        run_single_experiment(args.start_run, base_config, should_visualize=should_visualize)
    else:
        for run_id in range(args.start_run, args.start_run + args.num_runs):
            run_single_experiment(run_id, base_config, should_visualize=should_visualize)


if __name__ == '__main__':
    main()