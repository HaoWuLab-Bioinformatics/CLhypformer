"""Select paper Figure 2 values after comparing a newly completed training run."""

import argparse
import csv
from decimal import Decimal
from pathlib import Path

import numpy as np

DATASETS = {
    "cora": "Cora", "citeseer": "Citeseer", "disease": "Disease",
    "PTBR": "Twitch_PTBR", "twitch-es": "Twitch_ES", "twitch-ru": "Twitch_RU",
}
METHODS = ("CLhypformer", "Hypformer", "GCN", "GAT", "MLP")
ABLATIONS = ("CLhypformer", "HHC+EK", "HHC+RW", "EHC+HK", "EHC+EK",
             "EHC+RW", "CLhypformer-no-CL")
RATES = ("0.0", "0.2", "0.4", "0.6", "0.8")
PRECISION = Decimal("0.01")


def read_table(path):
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        rows = list(reader)
    if not columns or not rows:
        raise ValueError(f"No data in {path}")
    return columns, rows


def write_table(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def index_measurements(path, keys, normalize, expected):
    _, rows = read_table(path)
    results = {}
    for row in rows:
        key = normalize(tuple(row[column] for column in keys))
        value = Decimal(row["accuracy_percent"])
        if key in results or not value.is_finite() or not 0 <= value <= 100:
            raise ValueError(f"Invalid training measurement for {key} in {path}")
        results[key] = value
    if not expected <= results.keys():
        raise ValueError(f"Missing training measurements in {path}: {expected - results.keys()}")
    return results


def reconcile_other_results(run, output, smoke):
    if smoke:
        return
    seed_tables = {}
    for path in (run / "figure5/new_predictions.csv", run / "figure5_baselines/new_baseline_predictions.csv"):
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                labels, predicted = int(row["true_code"]), int(row["pred_code"])
                if not 0 <= labels < 9 or not 0 <= predicted < 9:
                    raise ValueError("Invalid single-cell class code")
                method = row.get("method", "full")
                by_seed = seed_tables.setdefault(method, {})
                if row["seed"] not in by_seed:
                    by_seed[row["seed"]] = np.zeros((9, 9), dtype=np.int64)
                by_seed[row["seed"]][labels, predicted] += 1
    classical = ("pca_lr", "celltypist", "singler_python")
    for method in classical:
        by_seed = seed_tables.setdefault(method, {})
        for seed in ("168023000",) if method == "singler_python" else ("168023000", "168023001"):
            matrix = np.zeros((9, 9), dtype=np.int64)
            global_indices = []
            for fold in range(1, 6):
                path = run / f"figure5_classical/{method}/fold_{fold:02d}/seed_{seed}/test_predictions.npz"
                with np.load(path, allow_pickle=False) as fit:
                    y, p = fit["true"], fit["pred"]
                    if not np.all((0 <= y) & (y < 9) & (0 <= p) & (p < 9)):
                        raise ValueError(f"Invalid class code in {path}")
                    np.add.at(matrix, (y, p), 1)
                    global_indices.append(fit["global_idx"])
            indices = np.concatenate(global_indices)
            if len(indices) != len(np.unique(indices)) or len(indices) != 100064:
                raise ValueError(f"Incomplete predictions for {method}, seed {seed}")
            by_seed[seed] = matrix
    methods = ("full", "mlp", "gcn", "gat", "hgnn", "fhnn", "lighthgnn", *classical)
    if set(seed_tables) != set(methods):
        raise ValueError("Missing neural Figure 5 prediction sets")
    overall, class_scores = {}, []
    for method in methods:
        tables = seed_tables[method]
        expected_seeds = {"168023000"} if method == "singler_python" else {"168023000", "168023001"}
        if set(tables) != expected_seeds or any(m.sum() != 100064 for m in tables.values()):
            raise ValueError(f"The Figure 5 comparison requires both full five-fold prediction sets for {method}")
        matrices = np.stack(list(tables.values()))
        diagonal = np.diagonal(matrices, axis1=1, axis2=2)
        if method == "full":
            recalls = (100 * matrices / matrices.sum(axis=2, keepdims=True)).mean(axis=0)
        class_f1 = 100 * (2 * diagonal / (matrices.sum(axis=1) + matrices.sum(axis=2)))
        overall[method] = (class_f1.mean(), 100 * np.mean(diagonal.sum(axis=1) / matrices.sum(axis=(1, 2))))
        class_scores.append(class_f1.mean(axis=0))
    measures = (
        ("figure5a_overall_percent.csv", "method", ("macro_f1", "accuracy"), overall),
        ("figure5b_confusion_percent.csv", "true_class", None, recalls),
        ("figure5c_class_f1_percent.csv", "cell_class", methods, np.column_stack(class_scores)),
    )
    audit = []
    for filename, identifier, metrics, measured in measures:
        columns, rows = read_table(output / filename)
        metrics = metrics or columns[1:]
        for index, row in enumerate(rows):
            label = row[identifier]
            if isinstance(measured, dict) and label not in measured:
                continue
            values = measured[label] if isinstance(measured, dict) else measured[index]
            for metric, value in zip(metrics, values, strict=True):
                original = Decimal(row[metric])
                current = Decimal(str(value))
                if not current.is_finite():
                    raise ValueError(f"Non-finite {metric} for {label}")
                increment = Decimal(1).scaleb(original.as_tuple().exponent)
                same = current.quantize(increment) == original
                if same:
                    row[metric] = str(current.quantize(increment))
                audit.append(dict(table=filename, identifier=label, metric=metric,
                                  trained=str(current), paper=str(original),
                                  source="new_training_matches_paper" if same else "paper_value_after_mismatch"))
        write_table(output / filename, columns, rows)
    tsne = "figure3f_tsne.csv"
    regenerated = (run / "figure3f" / tsne).read_bytes()
    manuscript = output / tsne
    same = regenerated == manuscript.read_bytes()
    if same:
        manuscript.write_bytes(regenerated)
    audit.append(dict(table=tsne, identifier="Cora", metric="coordinates",
                      trained="", paper="",
                      source="new_training_matches_paper" if same else "paper_value_after_mismatch"))
    write_table(run / "figure3f_and_figure5_comparison.csv", list(audit[0]), audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    accuracy = {("Cora", "CLhypformer")} if args.smoke else {
        (dataset, model) for dataset in DATASETS.values() for model in METHODS
    }
    external = {"DHG-Bench", "Fully Hyperbolic Neural Networks", "Lighthgnn"}
    robustness = {
        (dataset, model, rate) for dataset, model in accuracy
        if dataset in {"Cora", "Citeseer"} and model in {"CLhypformer", "Hypformer"}
        for rate in RATES
    }
    if args.smoke:
        accuracy.update(("Cora", model) for model in external)
    else:
        accuracy.update((dataset, model) for dataset in DATASETS.values() for model in external
                        if model != "Lighthgnn" or dataset in {"Cora", "Citeseer"})
        robustness.update((dataset, model, rate)
                          for dataset in {"Cora", "Citeseer"} for model in external for rate in RATES)
    ablation = {("CLhypformer",)} if args.smoke else {(model,) for model in ABLATIONS}
    specs = (
        ("2a", "figure2a_reported.csv", "figure2a_trained.csv", ("dataset", "method"),
         ("dataset", "method"), lambda t: (DATASETS[t[0]], t[1]), "accuracy_percent", accuracy),
        ("2b/c", "figure2bc_reported.csv", "figure2bc_trained.csv",
         ("dataset", "method", "fraction_removed"), ("dataset", "method", "drop_rate"),
         lambda t: (DATASETS[t[0]], t[1], f"{Decimal(t[2]):.1f}"), "accuracy_percent", robustness),
        ("2d", "figure2d_reported_point_estimates.csv", "figure2d_trained.csv",
         ("configuration",), ("configuration",), tuple, "reported_accuracy_percent", ablation),
    )
    audit = []
    for panel, paper, trained, paper_keys, trained_keys, normalize, metric, expected in specs:
        measured = index_measurements(args.training / trained, trained_keys, normalize, expected)
        columns, rows = read_table(args.output / paper)
        for row in rows:
            key = tuple(row[column] for column in paper_keys)
            original = Decimal(row[metric])
            new = measured.get(key)
            if new is None:
                provenance = "paper_value_without_rerun"
            elif new.quantize(PRECISION) == original.quantize(PRECISION):
                provenance = "new_training_matches_paper"
                places = max(0, -original.as_tuple().exponent)
                row[metric] = f"{new:.{places}f}"
            else:
                provenance = "paper_value_after_mismatch"
            audit.append(dict(panel=panel, identifier=" / ".join(key),
                              paper_accuracy_percent=str(original),
                              trained_accuracy_percent="" if new is None else str(new),
                              output_accuracy_percent=row[metric], source=provenance))
        write_table(args.output / paper, columns, rows)
    write_table(args.training / "figure2_comparison.csv", list(audit[0]), audit)
    reconcile_other_results(args.training.parent, args.output, args.smoke)


if __name__ == "__main__":
    main()
