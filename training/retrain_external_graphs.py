"""Run the external Figure 2 trainers and append their fresh measurements."""

import argparse
import csv
from pathlib import Path
import statistics
import subprocess
import sys

WORKER = Path(__file__).with_name("retrain_graph_baseline.py")
DATASETS = ("cora", "citeseer", "disease", "PTBR", "twitch-es", "twitch-ru")
METHODS = ("DHG-Bench", "Fully Hyperbolic Neural Networks", "Lighthgnn")


def load_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append(path, fieldnames, rows):
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Existing Figure 2 training directory")
    parser.add_argument("--light-python", default=sys.executable)
    parser.add_argument("--smoke", action="store_true")
    cfg = parser.parse_args()
    scores, robust = [], []
    expected_runs = 1 if cfg.smoke else 10
    for model in METHODS:
        # The manuscript values are retained for the four LightHGNN experiments
        # whose historical real-data inputs have not been recovered.
        datasets = ("cora", "citeseer") if model == "Lighthgnn" else DATASETS
        python = cfg.light_python if model == "Lighthgnn" else sys.executable
        for dataset in (("cora",) if cfg.smoke else datasets):
            job = cfg.output.resolve() / "external" / model / dataset
            job.parent.mkdir(parents=True, exist_ok=True)
            log = job.with_suffix(".log")
            print(f"Training {model} on {dataset}. Log: {log}", flush=True)
            command = [python, "-u", str(WORKER.resolve()), "--model", model, "--dataset", dataset,
                       "--sources", str(cfg.sources.resolve()), "--output", str(job)]
            if cfg.smoke:
                command.append("--smoke")
            with log.open("w", encoding="utf-8") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            values = load_rows(job / "accuracy.csv")
            if len(values) != expected_runs:
                raise ValueError(f"{model} on {dataset}: expected {expected_runs} runs")
            scores.append(dict(dataset=dataset, method=model,
                               accuracy_percent=statistics.mean(float(r["accuracy_percent"]) for r in values)))
            if dataset in {"cora", "citeseer"} and not cfg.smoke:
                rows = load_rows(job / "robustness.csv")
                if {float(r["drop_rate"]) for r in rows} != {0, 0.2, 0.4, 0.6, 0.8}:
                    raise ValueError(f"Incomplete masking results for {model} on {dataset}")
                robust.extend(rows)
    append(cfg.output / "figure2a_trained.csv", ("dataset", "method", "accuracy_percent"), scores)
    if robust:
        append(cfg.output / "figure2bc_trained.csv", ("dataset", "method", "drop_rate", "accuracy_percent"), robust)


if __name__ == "__main__":
    main()
