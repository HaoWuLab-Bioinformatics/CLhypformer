"""Run one external graph baseline from its original training implementation."""

import argparse
import csv
import importlib
from pathlib import Path
import json
import os
import statistics
import sys

DATASETS = ("cora", "citeseer", "disease", "PTBR", "twitch-es", "twitch-ru")
RATES = (0.0, 0.2, 0.4, 0.6, 0.8)
FH_DATASETS = {"disease": "disease_nc", "PTBR": "twitch-PTBR",
               "twitch-es": "twitch-ES", "twitch-ru": "twitch-RU"}
DHG_DATASETS = {"disease": "disease_nc", "PTBR": "twitch_ptbr",
                "twitch-es": "twitch_ES", "twitch-ru": "twitch_RU"}
LIGHT_SPLITS = {"cora": (20, 100), "citeseer": (20, 100), "disease": (20, 100),
                "PTBR": (200, 300), "twitch-es": (150, 250), "twitch-ru": (150, 250)}


def fhnn(args):
    from argparse import Namespace
    import torch

    source = args.sources / "fully-hyperbolic-nn-main/gcn"
    sys.path[:0] = [str(source), str(source.parent)]
    os.environ["DATAPATH"] = str(source / "data/raw")
    trainer = importlib.import_module("run_train")
    trainer.train.__globals__["format_metrics"] = lambda metrics, split: " ".join(
        f"{split}_{key}: {float(value):.4f}" for key, value in metrics.items())
    dataset = FH_DATASETS.get(args.dataset, args.dataset)
    default = trainer.get_base_config()
    config = dict(default, dataset=dataset, cuda=args.device)
    if args.dataset in {"cora", "citeseer", "disease"}:
        config.update(epochs=2000, patience=500, dim=16, num_layers=3, weight_decay=0.01)
        if args.dataset == "disease":
            config.update(lr=0.0001, dropout=0.2)
        else:
            config.update(lr=0.02, dropout=0.9)
    if args.smoke:
        config.update(epochs=5, min_epochs=1)
    clean_scores = []
    robust_scores = {rate: [] for rate in RATES}
    for seed in range(1234, 1234 + (1 if args.smoke else 10)):
        fitted = trainer.run_single_experiment(seed - 1234,
            dict(config, save_dir=str(args.output / f"seed_{seed}")), should_visualize=False)
        if not fitted["success"]:
            raise RuntimeError(fitted["error"])
        clean_scores.append(float(fitted["test_metrics"]["acc"]))
        if args.dataset in {"cora", "citeseer"}:
            from models.base_models import NCModel
            from utils.data_utils import load_data

            folder = Path(fitted["save_dir"])
            options = Namespace(**json.loads((folder / "config.json").read_text()))
            data = load_data(options, str(source / "data/raw" / dataset))
            options.n_nodes, options.feat_dim = data["features"].shape
            options.data = data
            model = NCModel(options).to(options.device)
            model.load_state_dict(torch.load(folder / "model.pth", map_location=options.device))
            model.eval()
            x = data["features"].to(options.device)
            test = data["idx_test"]
            truth = data["labels"][test].to(options.device)
            with torch.no_grad():
                for rate in RATES:
                    if rate == 0:
                        adj = data["adj_train_norm"]
                    else:
                        masked = trainer.edge_mask_adj(data["adj_train"], rate, seed=seed+100)
                        adj = trainer.recompute_adj_norm(masked)
                    adj = adj.to(options.device)
                    pred = model.decode(model.encode(x, adj), adj, test).argmax(dim=-1)
                    robust_scores[rate].append(float((pred == truth).float().mean().item()))
    robust = {rate: statistics.mean(scores) for rate, scores in robust_scores.items() if scores}
    return clean_scores, robust


def dhg(args):
    import numpy as np

    repo = args.sources / "DHG-Bench-main"
    sys.path.insert(0, str(repo / "dhgbench"))
    dataset = DHG_DATASETS.get(args.dataset, args.dataset)
    root = args.output / "data/data"
    cache = root / "pyg_data/hypergraph_dataset_updated"
    cache.mkdir(parents=True)
    (cache / dataset).symlink_to(repo / "dhgbench/data/data/pyg_data/hypergraph_dataset_updated" / dataset,
                                target_is_directory=True)
    for kind in ("trad_data", "hete_data"):
        (root / kind).symlink_to(repo / "data/data" / kind, target_is_directory=True)
    os.chdir(args.output)
    trainer = importlib.import_module("train")
    from lib_dataset.data_perturbation import perturbation

    robust_scores = {rate: [] for rate in RATES}
    original_agent = trainer.ExpAgent

    class RobustnessAgent(original_agent):
        def running(self, task_type, data):
            evaluate = self.evaluator.evaluate

            def with_masks(model, original, *, seed_split, task_type, verbose):
                clean = evaluate(model, original, seed_split=seed_split, task_type=task_type, verbose=verbose)
                for rate in RATES:
                    result = clean
                    if rate:
                        masked = perturbation(original, mode="drop_incidence", p=rate)
                        result = evaluate(model, masked, seed_split=seed_split, task_type=task_type, verbose=False)
                    robust_scores[rate].append(float(result["acc"][-1]))
                return clean

            self.evaluator.evaluate = with_masks
            super().running(task_type, data)

    if args.dataset in {"cora", "citeseer"}:
        trainer.ExpAgent = RobustnessAgent

    def export(embeddings, labels, **kwargs):
        path = args.output / "cora_embedding.npz"
        if args.dataset == "cora" and not path.exists():
            np.savez_compressed(path, points=embeddings.detach().cpu().numpy(),
                                labels=np.asarray(labels.detach().cpu() if hasattr(labels, "detach") else labels))

    trainer.visualize_embeddings = export
    config = dict(dataset=dataset, model="HGNN", task="node_cls", device=f"cuda:{args.device}",
                  num_runs=1 if args.smoke else 10, epochs=3 if args.smoke else 100,
                  train_ratio=0.5, val_ratio=0.25, lr=0.001, dropout=0.5,
                  All_num_layers=2, MLP_hidden=128)

    def fit(options):
        return [float(trainer.train_single_run(options, i)) / 100 for i in range(1, config["num_runs"]+1)]

    clean_scores = fit(dict(config, is_perturbed=False))
    robust = {rate: statistics.mean(scores) for rate, scores in robust_scores.items() if scores}
    robust = {rate: score / 100 for rate, score in robust.items()}
    return clean_scores, robust


def light(args):
    import numpy as np
    import torch

    torch.set_num_threads(2)
    source = args.sources / "LightHGNN-main"
    sys.path.insert(0, str(source))
    from omegaconf import OmegaConf

    trainer = importlib.import_module("train")
    dataset = DHG_DATASETS.get(args.dataset, args.dataset).lower()
    if args.dataset not in {"cora", "citeseer"}:
        raise RuntimeError("LightHGNN needs the original real-data loader for Disease and Twitch")
    cache = args.sources / ".dhg/datasets"
    if (cache / dataset).is_dir():
        from dhg.data import Cora, Citeseer

        def bundled_data(name, seed=None):
            data = {"cora": Cora, "citeseer": Citeseer}[name](data_root=str(cache))
            return data, data["edge_list"]

        trainer.load_data = bundled_data
    train, valid = LIGHT_SPLITS[args.dataset]
    cfg = OmegaConf.create(dict(
        data=dict(name=dataset, num_train=train, num_val=valid, ft_noise_level=0.0, hc_noise_level=1.0),
        model=dict(teacher="hgnn", student="light_hgnnp", hid=128),
        loss=dict(lamb=0, tau=1.0), experiment=dict(num_runs=1 if args.smoke else 10,
                                                  base_seed=2023, robustness_trials=5),
    ))
    clean_scores = []
    robust_scores = {rate: [] for rate in RATES}

    def evaluate_masking(model, features, graph, labels, test_mask, evaluator, seed):
        model.eval()
        for rate in RATES:
            rng = np.random.RandomState(seed)
            local = trainer._perturb_hypergraph_structure(graph, rate, rng=rng).to(features.device) if rate else graph
            values = trainer.test_with_mask(model, features, local, labels, test_mask, evaluator, 0)
            robust_scores[rate].append(float(values["accuracy"]))

    for seed in range(2023, 2023 + cfg.experiment.num_runs):
        fit = trainer.single_run(seed, cfg, evaluate_masking=evaluate_masking)
        clean_scores.append(float(fit["t"]["accuracy"]))
        if args.dataset == "cora" and seed == 2023:
            np.savez_compressed(args.output / "cora_embedding.npz",
                                points=fit["emb_t"].cpu().numpy(), labels=fit["labels"].cpu().numpy())
    robust = {rate: statistics.mean(scores) for rate, scores in robust_scores.items()}
    return clean_scores, robust


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    methods = {"Fully Hyperbolic Neural Networks": fhnn, "DHG-Bench": dhg, "Lighthgnn": light}
    parser.add_argument("--model", choices=methods, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.sources = args.sources.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    scores, robustness = methods[args.model](args)
    if any(not 0 <= v <= 1 for v in (*scores, *robustness.values())):
        raise ValueError("An external trainer returned an invalid fraction")
    with (args.output / "accuracy.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "run", "accuracy_percent"])
        writer.writerows((args.dataset, args.model, i, value*100) for i, value in enumerate(scores, 1))
    if robustness:
        with (args.output / "robustness.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "method", "drop_rate", "accuracy_percent"])
            writer.writerows((args.dataset, args.model, rate, 100 * value)
                             for rate, value in sorted(robustness.items()))


if __name__ == "__main__":
    main()
