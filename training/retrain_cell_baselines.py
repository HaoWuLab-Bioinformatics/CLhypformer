"""Retrain the neural Figure 5 comparators and export held-out predictions."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

WORKER = Path(__file__).with_name("gse_caf_confirm_worker_20261022.py")
METHODS = ("mlp", "gcn", "gat", "hgnn", "fhnn", "lighthgnn")
SEEDS = (168023000, 168023001)
SCALES = (1.0, 0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", required=True, type=Path)
    parser.add_argument("--baselines-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--light-python", default=sys.executable)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--smoke", action="store_true")
    cfg = parser.parse_args()
    cfg.output = cfg.output.resolve()
    cfg.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CLHYP_ATLAS_ROOT=str(cfg.atlas_root.resolve()),
               CLHYP_BASELINES_ROOT=str(cfg.baselines_root.resolve()), PYTHONDONTWRITEBYTECODE="1")
    output_rows = []
    for method in dict.fromkeys(cfg.methods):
        python = cfg.light_python if method == "lighthgnn" else sys.executable
        method_rows = []
        for fold in ((1,) if cfg.smoke else range(1, 6)):
            for seed in (SEEDS[:1] if cfg.smoke else SEEDS):
                case = cfg.output / method / f"fold_{fold:02d}" / f"seed_{seed}"
                case.mkdir(parents=True)
                base = [python, "-u", str(WORKER.resolve()), "--method", method,
                        "--fold", str(fold), "--seed", str(seed)]

                def launch(destination, arguments):
                    destination.mkdir(exist_ok=True)
                    command = [*base, "--output-dir", str(destination), *arguments]
                    (destination / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
                    log_path = destination / "train.log"
                    print(f"Running {method}, fold {fold}, seed {seed}. Log: {log_path}", flush=True)
                    with log_path.open("w", encoding="utf-8") as log:
                        subprocess.run(command, cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       check=True)

                if cfg.smoke:
                    selected = case
                    launch(selected, ["--smoke", "--lr-scale", "0.2"])
                else:
                    candidates = []
                    for scale in SCALES:
                        candidate = case / f"lr_{scale:g}"
                        launch(candidate, ["--train-only", "--lr-scale", str(scale)])
                        result = json.loads((candidate / "complete.json").read_text(encoding="utf-8"))
                        result["checkpoint"] = str(candidate / "checkpoint.pt")
                        candidates.append(result)
                    best = max(candidates, key=lambda r: (r["validation_macro_f1"], r["lr_scale"]))
                    selected = case / "selected"
                    selected.mkdir()
                    selection = dict(best, alternatives=candidates, test_used_for_selection=False)
                    (selected / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
                    launch(selected, ["--export-only"])
                    if not (selected / "export_complete.json").is_file():
                        raise RuntimeError(f"Missing evaluation record for {selected}")
                with np.load(selected / "test.npz", allow_pickle=False) as result:
                    method_rows.extend((method, fold, seed, int(i), int(y), int(p))
                                       for i, y, p in zip(result["idx"], result["true"], result["pred"], strict=True))
        if not cfg.smoke:
            for seed in SEEDS:
                indices = [r[3] for r in method_rows if r[2] == seed]
                if len(indices) != len(set(indices)) or len(indices) != 100064:
                    raise ValueError(f"Incomplete held-out test coverage for {method}, seed {seed}")
        output_rows.extend(method_rows)
    with (cfg.output / "new_baseline_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "fold", "seed", "global_idx", "true_code", "pred_code"])
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
