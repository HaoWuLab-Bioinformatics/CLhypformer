"""Replay logged CLhypformer classifier fits against the frozen processed atlas."""
import argparse
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = Path("bio_runs/gse176078_full_20261017")
SEEDS = (168023000, 168023001)


def value(args, key, replacement):
    args[args.index(key) + 1] = str(replacement)


def load_command(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True,
                        help="Project containing medium/ and the frozen bio_runs/ tree")
    parser.add_argument("--output", type=Path, default=HERE.parent / "retrained_full")
    parser.add_argument("--device", type=int, default=0, help="CUDA index among visible devices")
    parser.add_argument("--smoke", action="store_true", help="One fold, one seed, two epochs; omit for 10 fits")
    cfg = parser.parse_args()
    import torch
    if cfg.device < 0 or cfg.device >= torch.cuda.device_count():
        parser.error("This archived trainer requires an available CUDA device")
    project = cfg.project.resolve()
    out = cfg.output.resolve()
    medium = project / "medium"
    runner = medium / "gse176078_full_eval.py"
    if not runner.is_file():
        parser.error(f"Missing training source: {runner}")
    if out.is_relative_to(project / "bio_runs"):
        parser.error("Choose an output directory outside the frozen bio_runs tree")
    cases = [(fold, seed) for fold in range(1, 6) for seed in SEEDS]
    if cfg.smoke:
        cases = cases[:1]
        out = out / "smoke"
    planned = []
    checksums = load_command(HERE / "frozen_input_sha256.json")
    verified = set()
    for fold, seed in cases:
        rel = Path(f"fold_{fold:02d}/seed_{seed}")
        flags = load_command(HERE / "commands" / rel / "flags.json")
        data = project / BASE / f"data/fold_{fold:02d}/biomed/gse176078_brca.npz"
        prior = project / BASE / f"state/fold_{fold:02d}/biostate.npz"
        for path in (data, prior):
            if not path.is_file():
                parser.error(f"Missing frozen processed input: {path}")
            if path not in verified:
                with path.open("rb") as f:
                    digest = hashlib.file_digest(f, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(f.read()).hexdigest()
                if digest != checksums[path.relative_to(project).as_posix()]:
                    parser.error(f"Frozen input checksum differs from original: {path}")
                verified.add(path)
        target = out / rel
        for key, val in (("--state-data", prior), ("--npz", data), ("--output-dir", target),
                         ("--device", cfg.device)):
            value(flags, key, val)
        if cfg.smoke:
            value(flags, "--epochs", 2)
            value(flags, "--min-checkpoint-epoch", 1)
        planned.append((fold, seed, data, target, [sys.executable, "-u", str(runner), *flags]))
    rows = []
    for fold, seed, data, target, cmd in planned:
        if target.exists():
            raise FileExistsError(f"Run directory already exists: {target}. Choose a fresh --output.")
        target.mkdir(parents=True)
        (target / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
        print(f"Training fold {fold}, seed {seed}; log: {target / 'train.log'}", flush=True)
        start = time.monotonic()
        with (target / "train.log").open("w", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=medium, env={**os.environ, "PYTHONHASHSEED": str(seed),
                           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                           "PYTHONDONTWRITEBYTECODE": "1"},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        with np.load(target / "predictions.npz", allow_pickle=False) as pred, np.load(data, allow_pickle=False) as z:
            idx = pred["test_idx"]
            predicted = pred["prob"].argmax(1)
            truth = z["labels"][idx]
            if not np.array_equal(np.sort(idx), np.sort(z["test_idx"])) or not np.isfinite(pred["prob"]).all():
                raise ValueError(f"Invalid test prediction file: {target}")
        elapsed = time.monotonic() - start
        rows.extend((fold, seed, int(i), int(t), int(p)) for i, t, p in zip(idx, truth, predicted, strict=True))
        print(f"Completed fold {fold}, seed {seed} in {elapsed:.1f}s", flush=True)
    if not cfg.smoke:
        for seed in SEEDS:
            indices = [row[2] for row in rows if row[1] == seed]
            if len(indices) != len(set(indices)) or len(indices) != 100064:
                raise ValueError(f"Incomplete pooled fold coverage: {seed}")
    with (out / "new_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "seed", "global_idx", "true_code", "pred_code"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
