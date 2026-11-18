#!/usr/bin/env python
"""
消融实验运行脚本
系统地变换三个维度各两种配置，共 2x2x2 = 8 组实验:
  (1) cl_space:      对比学习计算空间  hyp / euc
  (2) knn_metric:    建图度量空间      hyp / euc
  (3) graph_build_method: 建图算法     knn / random_walk

用法:
  python ablation_runner.py
  python ablation_runner.py --dataset cora --epochs 100
"""

import subprocess
import sys
import os
import itertools
import argparse
import time
from datetime import datetime


def get_ablation_configs():
    """返回所有消融实验的配置组合"""
    configs = [
        ("hyp", "hyp", "knn",         "CLHypformer (原始配置)"),
        ("hyp", "hyp", "random_walk", "消融: 随机游走建图"),
        ("hyp", "euc", "knn",         "消融: 欧式KNN建图"),
        ("hyp", "euc", "random_walk", "消融: 欧式随机游走建图"),
        ("euc", "hyp", "knn",         "消融: 欧式CL空间"),
        ("euc", "hyp", "random_walk", "消融: 欧式CL+随机游走"),
        ("euc", "euc", "knn",         "消融: 欧式CL+欧式KNN"),
        ("euc", "euc", "random_walk", "消融: 全欧式"),
    ]
    return configs


def build_command(args, cl_space, knn_metric, graph_build_method, run_id):
    """构建 main.py 的命令行参数"""
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"),
        f"--dataset={args.dataset}",
        f"--method={args.method}",
        f"--hidden_channels={args.hidden_channels}",
        f"--epochs={args.epochs}",
        f"--runs={args.runs}",
        f"--lr={args.lr}",
        f"--hyp_lr={args.hyp_lr}",
        f"--weight_decay={args.weight_decay}",
        f"--hyp_weight_decay={args.hyp_weight_decay}",
        f"--patience={args.patience}",
        f"--device={args.device}",
        f"--rand_split_class={args.rand_split_class}",
        f"--label_num_per_class={args.label_num_per_class}",
        f"--valid_num={args.valid_num}",
        f"--test_num={args.test_num}",
        # 消融实验关键参数
        f"--cl_space={cl_space}",
        f"--knn_metric={knn_metric}",
        f"--graph_build_method={graph_build_method}",
        # 模型参数
        "--use_cl=1",
        "--do_pretrain=1",
        f"--cl_weight={args.cl_weight}",
        f"--cl_knn={args.cl_knn}",
        f"--cl_drop_rate={args.cl_drop_rate}",
        f"--cl_tau={args.cl_tau}",
        "--use_adaptive_curvature=1",
        "--drop_strategy=critical",
        f"--use_graph={args.use_graph}",
        f"--graph_weight={args.graph_weight}",
        f"--k_in={args.k_in}",
        f"--k_hidden={args.k_hidden}",
        f"--k_out={args.k_out}",
        f"--power_k={args.power_k}",
        f"--attention_type={args.attention_type}",
        f"--decoder_type={args.decoder_type}",
        f"--trans_num_heads={args.trans_num_heads}",
        f"--gnn_num_layers={args.gnn_num_layers}",
        f"--trans_num_layers={args.trans_num_layers}",
        f"--display_step={args.display_step}",
        f"--run_id={run_id}",
    ]
    if args.no_feat_norm:
        cmd.append(f"--no_feat_norm={args.no_feat_norm}")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="消融实验运行器")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hyp_lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--hyp_weight_decay", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--cl_weight", type=float, default=0.1)
    parser.add_argument("--cl_knn", type=int, default=5)
    parser.add_argument("--cl_drop_rate", type=float, default=0.5)
    parser.add_argument("--cl_tau", type=float, default=0.5)
    parser.add_argument("--use_graph", type=int, default=1)
    parser.add_argument("--graph_weight", type=float, default=0.8)
    parser.add_argument("--k_in", type=float, default=0.1)
    parser.add_argument("--k_hidden", type=float, default=0.1)
    parser.add_argument("--k_out", type=float, default=0.5)
    parser.add_argument("--power_k", type=float, default=1.0)
    parser.add_argument("--attention_type", type=str, default="linear_focused")
    parser.add_argument("--decoder_type", type=str, default="hyp")
    parser.add_argument("--method", type=str, default="hypformer")
    parser.add_argument("--trans_num_heads", type=int, default=4)
    parser.add_argument("--gnn_num_layers", type=int, default=2)
    parser.add_argument("--trans_num_layers", type=int, default=1)
    parser.add_argument("--rand_split_class", type=int, default=1)
    parser.add_argument("--label_num_per_class", type=int, default=20)
    parser.add_argument("--valid_num", type=int, default=500)
    parser.add_argument("--test_num", type=int, default=1000)
    parser.add_argument("--no_feat_norm", type=int, default=1)
    parser.add_argument("--display_step", type=int, default=5)
    args = parser.parse_args()

    configs = get_ablation_configs()
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "ablation", args.dataset,
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(f"  消融实验: {args.dataset}")
    print(f"  共 {len(configs)} 组实验")
    print(f"  结果保存: {results_dir}")
    print("=" * 80)

    results = {}
    for idx, (cl_space, knn_metric, graph_build, desc) in enumerate(configs):
        run_id = f"ablation_{idx:02d}"
        print(f"\n{'=' * 60}")
        print(f"  实验 {idx + 1}/{len(configs)}: {desc}")
        print(f"    cl_space={cl_space}, knn_metric={knn_metric}, graph_build={graph_build}")
        print(f"{'=' * 60}")

        cmd = build_command(args, cl_space, knn_metric, graph_build, run_id)
        start_time = time.time()
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        elapsed = time.time() - start_time

        status = "SUCCESS" if result.returncode == 0 else f"FAILED (code={result.returncode})"
        results[(cl_space, knn_metric, graph_build)] = {
            "description": desc,
            "status": status,
            "time": elapsed
        }
        print(f"  -> {status} (耗时: {elapsed:.0f}s)")

    # 保存汇总结果
    summary_path = os.path.join(results_dir, "ablation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"消融实验汇总: {args.dataset}\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'#':<4} {'CL空间':<8} {'建图度量':<10} {'建图算法':<12} {'描述':<30} {'状态':<20} {'耗时':<10}\n")
        f.write("-" * 100 + "\n")
        for idx, (cl_space, knn_metric, graph_build, desc) in enumerate(configs):
            r = results.get((cl_space, knn_metric, graph_build), {})
            f.write(f"{idx+1:<4} {cl_space:<8} {knn_metric:<10} {graph_build:<12} "
                    f"{desc:<30} {r.get('status', 'N/A'):<20} {r.get('time', 0):.0f}s\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("实验设计 (3个维度 x 2种配置 = 8组):\n")
        f.write("  (1) cl_space:      对比学习计算空间   hyp=双曲空间  euc=欧式空间\n")
        f.write("  (2) knn_metric:    建图度量空间       hyp=双曲距离  euc=欧式距离\n")
        f.write("  (3) graph_build:   建图算法           knn=K-近邻  random_walk=随机游走\n")

    print(f"\n{'=' * 60}")
    print("  消融实验完成!")
    print(f"  汇总结果: {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
