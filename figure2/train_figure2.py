"""Train Figure 2 configurations supported by the CLhypfomer source tree."""

import argparse
import csv
import json
import os
import random
import runpy
import secrets
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model"
DATASETS = ("cora", "citeseer", "disease", "PTBR", "twitch-es", "twitch-ru")
ABLATIONS = {
    "HHC+EK": ("hyp", "euc", "knn"),
    "HHC+RW": ("hyp", "euc", "random_walk"),
    "EHC+HK": ("euc", "hyp", "knn"),
    "EHC+EK": ("euc", "euc", "knn"),
    "EHC+RW": ("euc", "euc", "random_walk"),
    "CLhypformer-no-CL": ("hyp", "hyp", "knn"),
}
METHODS = ("CLhypformer", "Hypformer", "GCN", "GAT", "MLP")
RATES = (0.0, 0.2, 0.4, 0.6, 0.8)


def run_worker(path):
    import numpy as np
    import torch

    options = json.loads(path.read_text(encoding="utf-8"))
    seed = options["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(MODEL))
    import dataset

    dataset.DATAPATH = options["data_dir"]
    args = [f"--{key}={value}" for key, value in options.items()]
    if options["use_cl"]:
        args.append("--use_adaptive_curvature")
    sys.argv = [str(MODEL / "main.py"), *args]
    state = runpy.run_path(sys.argv[0], run_name="__main__")
    histories = state["logger"].results
    split_list = state["split_idx_lst"]
    robustness = state["all_robustness_results"]
    if len(histories) != options["runs"] or len(robustness) != options["runs"]:
        raise RuntimeError("Training did not yield an evaluated model for each requested run")
    rows = []
    for i, history in enumerate(histories):
        best_val_epoch = max(range(len(history)), key=lambda j: history[j][1])
        metrics = {
            "run": i + 1,
            "best_validation_epoch": best_val_epoch + 1,
            "highest_test_accuracy_percent": 100 * max(row[2] for row in history),
            "test_accuracy_at_best_validation_percent": 100 * history[best_val_epoch][2],
        }
        metrics.update({f"edge_drop_{int(rate * 100)}_percent": robustness[i]["edge"][rate]
                        for rate in RATES})
        rows.append(metrics)
    write_csv(Path("training_metrics.csv"), rows)
    splits = [{key: indices.cpu().tolist() for key, indices in split.items()} for split in split_list]
    Path("splits.json").write_text(json.dumps(splits), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_figure_tables(output, summary):
    accuracy, robustness, ablation = [], [], []
    for row in summary:
        dataset, method = row["dataset"], row["method"]
        peak = row["highest_test_accuracy_percent"]
        if method in METHODS:
            accuracy.append(dict(dataset=dataset, method=method, accuracy_percent=peak))
        if dataset in {"cora", "citeseer"} and method in {"CLhypformer", "Hypformer"}:
            for rate in RATES:
                score = peak if rate == 0.0 else row[f"edge_drop_{int(rate * 100)}_percent"]
                robustness.append(dict(dataset=dataset, method=method, drop_rate=rate,
                                       accuracy_percent=score))
        if dataset == "cora" and (method == "CLhypformer" or method in ABLATIONS):
            ablation.append(dict(configuration=method, accuracy_percent=peak))
    for name, rows in (("figure2a_trained.csv", accuracy), ("figure2bc_trained.csv", robustness),
                       ("figure2d_trained.csv", ablation)):
        if rows:
            write_csv(output / name, rows)


def configuration(dataset, method, args):
    options = {
        "dataset": dataset, "data_dir": str(args.data_dir), "seed": args.seed,
        "method": method.lower() if method in METHODS[2:] else "hypformer",
        "epochs": 2 if args.smoke else args.epochs,
        "runs": 1 if args.smoke else args.runs,
        "lr": 0.005, "hidden_channels": 256, "weight_decay": 0.005,
        "patience": 200, "gnn_num_layers": 3, "graph_weight": 0.8,
        "decoder_type": "euc", "power_k": 1.0, "k_in": 1.0, "k_out": 3.0,
        "rand_split": 1, "train_prop": 0.5, "valid_prop": 0.25,
        "cl_weight": 0.2, "cl_tau": 0.7, "cl_knn": 5, "cl_drop_rate": 0.5,
        "pretrain_epochs": 1 if args.smoke else 200,
        "use_cl": int(method == "CLhypformer" or method in ABLATIONS and method != "CLhypformer-no-CL"),
        "save_topology": 0, "device": args.device, "display_step": 25,
    }
    if dataset == "citeseer":
        options.update(gnn_num_layers=5, graph_weight=0.4, power_k=3.0, k_out=1.0)
    if dataset == "cora" and method in {"CLhypformer", "Hypformer"} and not args.smoke:
        options.update(save_topology=1, topology_epoch=13, topology_plot=0)
    if dataset == "disease":
        # The June Hypformer pair used the loader's 627-test-node split.
        # The August MLP log fits the 504-test-node class-stratified split.
        options["rand_split"] = int(method in {"CLhypformer", "Hypformer"})
    if dataset == "twitch-ru":
        options["drop_strategy"] = "critical"
    if method in ABLATIONS:
        space, metric, graph = ABLATIONS[method]
        options.update(cl_space=space, knn_metric=metric, graph_build_method=graph,
                       drop_strategy="critical")
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--methods", nargs="+", choices=(*METHODS, *ABLATIONS), default=METHODS)
    parser.add_argument("--include-ablation", action="store_true")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, help="Reuse a recorded seed for the same random split")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = secrets.randbits(32)
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be between 0 and 4294967295")
    args.data_dir = args.data_dir.resolve()
    if not args.data_dir.is_dir() or min(args.runs, args.epochs) < 1:
        parser.error("Specify an existing data directory and positive run and epoch counts")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)

    jobs = dict.fromkeys(
        (dataset, method)
        for dataset in args.datasets for method in args.methods
        if method not in ABLATIONS or dataset == "cora"
    )
    if args.include_ablation and "cora" in args.datasets:
        jobs.update(dict.fromkeys(("cora", method) for method in ABLATIONS))
    if not jobs:
        parser.error("No jobs selected. Ablation configurations are for Cora.")
    print(f"Random seed for this invocation: {args.seed}", flush=True)
    summary = []
    for dataset, method in jobs:
        work = args.output / dataset / method
        work.mkdir(parents=True)
        config = work / "config.json"
        config.write_text(json.dumps(configuration(dataset, method, args), indent=2), encoding="utf-8")
        print(f"Training {dataset} / {method}. Log: {work / 'train.log'}", flush=True)
        env = dict(os.environ, PYTHONHASHSEED=str(args.seed), OPENBLAS_NUM_THREADS="2")
        with (work / "train.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", __file__, "_worker", str(config)],
                           cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        with (work / "training_metrics.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        averages = {key: statistics.mean(float(row[key]) for row in rows)
                    for key in rows[0] if key not in {"run", "best_validation_epoch"}}
        summary.append(dict(dataset=dataset, method=method, runs=len(rows), **averages))
        write_csv(args.output / "training_summary.csv", summary)
    export_figure_tables(args.output, summary)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "_worker":
        run_worker(Path(sys.argv[2]))
    else:
        main()
