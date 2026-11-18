import csv
import os
from datetime import datetime


class ResultSaver:
    """用于保存训练结果和鲁棒性测试结果的类"""

    def __init__(self, dataset_name, save_dir='results'):
        self.dataset_name = dataset_name
        self.save_dir = save_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.filename = f"{dataset_name}_{timestamp}_{os.getpid()}.csv"

        self.full_path = os.path.join(save_dir, dataset_name)
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)

        self.file_path = os.path.join(self.full_path, self.filename)

        self._initialize_csv()

    def _initialize_csv(self):
        """初始化CSV文件并写入表头"""
        headers = [
            'Dataset',
            'Method',
            'Run',
            'Best_Train_Acc',
            'Best_Val_Acc',
            'Highest_Test_Acc',
            'Final_Train_Acc',
            'Final_Test_Acc',
            'Best_Train_F1',
            'Best_Val_F1',
            'Highest_Test_F1',
            'Final_Train_F1',
            'Final_Test_F1',
            'Use_CL',
            'Use_Adaptive_Curvature',
            'CL_Space',
            'Graph_Build_Method',
            'knn_metric',
            'drop_strategy',
            'Robustness_Edge_Drop_0%',
            'Robustness_Edge_Drop_20%',
            'Robustness_Edge_Drop_40%',
            'Robustness_Edge_Drop_60%',
            'Robustness_Edge_Drop_80%',
            'Robustness_Feat_Mask_0%',
            'Robustness_Feat_Mask_20%',
            'Robustness_Feat_Mask_40%',
            'Robustness_Feat_Mask_60%',
            'Robustness_Feat_Mask_80%',
            'Avg_Robustness_Edge',
            'Avg_Robustness_Feat',
            'Timestamp'
        ]

        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def save_run_result(self, run_idx, train_acc, val_acc, highest_test_acc,
                       final_train_acc=None, final_test_acc=None,
                       train_f1=None, val_f1=None, test_f1=None,
                       final_train_f1=None, final_test_f1=None,
                       robustness_edge=None, robustness_feat=None,
                       method='HyperbolicTransformer',
                       use_cl=False, use_adaptive_curvature=False,
                       cl_space='hyp', graph_build_method='knn',
                       knn_metric='hyp', drop_strategy='critical'):
        """
        保存单次运行的结果
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        avg_robustness_edge = 0.0
        avg_robustness_feat = 0.0

        if robustness_edge:
            avg_robustness_edge = sum(robustness_edge.values()) / len(robustness_edge)

        if robustness_feat:
            avg_robustness_feat = sum(robustness_feat.values()) / len(robustness_feat)

        row = [
            self.dataset_name,
            method,
            run_idx + 1,
            f"{train_acc:.2f}",
            f"{val_acc:.2f}",
            f"{highest_test_acc:.2f}",
            f"{final_train_acc:.2f}" if final_train_acc is not None else "N/A",
            f"{final_test_acc:.2f}" if final_test_acc is not None else "N/A",
            f"{train_f1:.4f}" if train_f1 is not None else "N/A",
            f"{val_f1:.4f}" if val_f1 is not None else "N/A",
            f"{test_f1:.4f}" if test_f1 is not None else "N/A",
            f"{final_train_f1:.4f}" if final_train_f1 is not None else "N/A",
            f"{final_test_f1:.4f}" if final_test_f1 is not None else "N/A",
            str(use_cl),
            str(use_adaptive_curvature),
            str(cl_space),
            str(graph_build_method),
            str(knn_metric),
            str(drop_strategy),
        ]

        drop_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
        for dr in drop_rates:
            if robustness_edge and dr in robustness_edge:
                row.append(f"{robustness_edge[dr]:.2f}")
            else:
                row.append("N/A")

        for dr in drop_rates:
            if robustness_feat and dr in robustness_feat:
                row.append(f"{robustness_feat[dr]:.2f}")
            else:
                row.append("N/A")

        row.extend([
            f"{avg_robustness_edge:.2f}",
            f"{avg_robustness_feat:.2f}",
            timestamp
        ])

        with open(self.file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"✓ Run {run_idx + 1} results saved to {self.file_path}")

    def save_summary(self, logger_results, all_robustness_results=None, args=None, f1_summary=None):
        """
        保存所有运行的汇总统计结果
        """
        summary_path = os.path.join(self.full_path, f"{self.dataset_name}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Method: {args.method if args else 'N/A'}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if args:
                f.write(f"Use_CL: {getattr(args, 'use_cl', False)}\n")
                f.write(f"Use_Adaptive_Curvature: {getattr(args, 'use_adaptive_curvature', False)}\n")
                f.write(f"CL Space: {getattr(args, 'cl_space', 'hyp')}\n")
                f.write(f"Graph Build Method: {getattr(args, 'graph_build_method', 'knn')}\n")
                f.write(f"KNN Metric: {getattr(args, 'knn_metric', 'N/A')}\n")
                f.write(f"Drop Strategy: {getattr(args, 'drop_strategy', 'N/A')}\n")
                f.write(f"EPS Threshold: {getattr(args, 'eps_threshold', 'N/A')}\n")
                f.write(f"CL kNN: {getattr(args, 'cl_knn', 'N/A')}\n")
                f.write(f"Hidden Channels: {getattr(args, 'hidden_channels', 'N/A')}\n")
                f.write(f"Learning Rate: {getattr(args, 'lr', 'N/A')}\n")
                f.write(f"Epochs: {getattr(args, 'epochs', 'N/A')}\n")
            
            f.write("=" * 80 + "\n\n")

            f.write("Overall Statistics:\n")
            f.write(logger_results + "\n\n")

            if f1_summary:
                f.write("F1 Score Summary:\n")
                f.write("-" * 80 + "\n")
                for key, (mean_val, std_val) in f1_summary.items():
                    f.write(f"  {key}: {mean_val:.4f} ± {std_val:.4f}\n")
                f.write("\n")

            if all_robustness_results:
                f.write("Robustness Test Summary:\n")
                f.write("-" * 80 + "\n")

                edge_drop_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
                feat_drop_rates = [0.0, 0.2, 0.4, 0.6, 0.8]

                f.write("\nEdge Deletion Robustness (Average across runs):\n")
                for dr in edge_drop_rates:
                    accs = []
                    for r in all_robustness_results:
                        if r is not None and 'edge' in r and dr in r['edge']:
                            accs.append(r['edge'][dr])
                    if accs:
                        avg_acc = sum(accs) / len(accs)
                        f.write(f"  Drop Rate {dr*100:2.0f}% -> Avg Test Acc: {avg_acc:.2f}%\n")
                    else:
                        f.write(f"  Drop Rate {dr*100:2.0f}% -> No data available\n")

                f.write("\nFeature Masking Robustness (Average across runs):\n")
                for dr in feat_drop_rates:
                    accs = []
                    for r in all_robustness_results:
                        if r is not None and 'feat' in r and dr in r['feat']:
                            accs.append(r['feat'][dr])
                    if accs:
                        avg_acc = sum(accs) / len(accs)
                        f.write(f"  Drop Rate {dr*100:2.0f}% -> Avg Test Acc: {avg_acc:.2f}%\n")
                    else:
                        f.write(f"  Drop Rate {dr*100:2.0f}% -> No data available\n")

        print(f"✓ Summary saved to {summary_path}")

    def calculate_and_save_averages(self, num_runs):
        """
        计算并保存所有轮次的平均值和标准差
        """
        import numpy as np
        
        if not os.path.exists(self.file_path):
            return
        
        highest_test_accs = []
        best_train_f1s = []
        best_val_f1s = []
        highest_test_f1s = []
        final_train_f1s = []
        final_test_f1s = []
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    highest_test_accs.append(float(row['Highest_Test_Acc']))
                except (ValueError, KeyError):
                    continue
                for lst, key in [
                    (best_train_f1s, 'Best_Train_F1'),
                    (best_val_f1s, 'Best_Val_F1'),
                    (highest_test_f1s, 'Highest_Test_F1'),
                    (final_train_f1s, 'Final_Train_F1'),
                    (final_test_f1s, 'Final_Test_F1'),
                ]:
                    try:
                        val = float(row[key])
                        lst.append(val)
                    except (ValueError, KeyError):
                        pass
        
        if len(highest_test_accs) == 0:
            return
        
        avg_highest_test = np.mean(highest_test_accs)
        std_highest_test = np.std(highest_test_accs)

        f1_summary = {}
        for name, lst in [
            ('Best Train F1', best_train_f1s),
            ('Best Val F1', best_val_f1s),
            ('Highest Test F1', highest_test_f1s),
            ('Final Train F1', final_train_f1s),
            ('Final Test F1', final_test_f1s),
        ]:
            if lst:
                f1_summary[name] = (np.mean(lst), np.std(lst))
        
        summary_file = os.path.join(self.full_path, f"{self.dataset_name}_averages.txt")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Total Runs: {num_runs}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Summary Statistics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Highest Test Acc - Mean ± Std: {avg_highest_test:.2f} ± {std_highest_test:.2f}\n\n")

            if f1_summary:
                f.write("F1 Score Summary:\n")
                f.write("-" * 80 + "\n")
                for name, (mean_val, std_val) in f1_summary.items():
                    f.write(f"  {name}: {mean_val:.4f} ± {std_val:.4f}\n")
                f.write("=" * 80 + "\n")
        
        print(f"\n{'='*60}")
        print(f"📊 Final Summary for {self.dataset_name}")
        print(f"{'='*60}")
        print(f"Highest Test Acc (Mean ± Std): {avg_highest_test:.2f} ± {std_highest_test:.2f}")
        if f1_summary:
            print(f"\nF1 Score Summary:")
            print(f"{'-'*60}")
            for name, (mean_val, std_val) in f1_summary.items():
                print(f"  {name}: {mean_val:.4f} ± {std_val:.4f}")
        print(f"{'='*60}")
        print(f"✓ Averages saved to {summary_file}")

        return f1_summary
