"""Train the standard and strict gene-held-out ablation classifiers."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

BASE = Path("bio_runs/gse176078_full_20261017")
SEEDS = (168023000, 168023001)
ARMS = {
    "local": ("removed", 0, False),
    "fixed_fusion": ("nonlinear", 0, True),
    "gated_fusion": ("nonlinear", 0, False),
    "full": ("nonlinear", 0.001, False),
}


def evaluate(project):
    sys.path.insert(0, str(project / "medium"))
    if os.environ.get("REVISION_FIXED_GATE") == "1":
        from gse_biostate_model import BiologicalStateHypergraph

        original = BiologicalStateHypergraph.__init__

        def initialize(self, *args, **kwargs):
            original(self, *args, **kwargs)
            self.gate.requires_grad_(False)
            self.state_gate.requires_grad_(False)

        BiologicalStateHypergraph.__init__ = initialize
    from gse176078_full_eval import main

    del sys.argv[1:3]
    main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--strict-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    cfg = parser.parse_args()
    cfg.project = cfg.project.resolve()
    cfg.output = cfg.output.resolve()
    protocols = {
        "standard": (cfg.project / BASE, ("fixed_fusion",) if cfg.smoke else tuple(ARMS)),
        "strict": (cfg.strict_input.resolve(), ("local",) if cfg.smoke else ("local", "full")),
    }
    for protocol, (inputs, arms) in protocols.items():
        state = inputs / "state/fold_01/biostate.npz"
        atlas = inputs / "data/fold_01/biomed/gse176078_brca.npz"
        for path in (state, atlas):
            if not path.is_file():
                raise FileNotFoundError(path)
        for method in arms:
            mode, weight, fixed = ARMS[method]
            for seed in (SEEDS[:1] if cfg.smoke else SEEDS):
                opts = Path(__file__).with_name("commands") / f"fold_01/seed_{seed}/flags.json"
                flags = json.loads(opts.read_text(encoding="utf-8"))
                dest = cfg.output / protocol / method / str(seed)
                params = {"--state-mode": mode, "--cl-weight": weight, "--state-data": state,
                          "--npz": atlas, "--output-dir": dest}
                if cfg.smoke:
                    params.update({"--epochs": 2, "--min-checkpoint-epoch": 1})
                for key, value in params.items():
                    flags[flags.index(key) + 1] = str(value)
                dest.mkdir(parents=True, exist_ok=False)
                cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "_evaluate", str(cfg.project), *flags]
                (dest / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
                log = dest / "train.log"
                print(f"Training {protocol}/{method}/{seed}. Log: {log}", flush=True)
                env = dict(os.environ, REVISION_FIXED_GATE=str(int(fixed)), PYTHONHASHSEED=str(seed),
                           PYTHONDONTWRITEBYTECODE="1")
                with log.open("w", encoding="utf-8") as f:
                    subprocess.run(cmd, cwd=dest, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
                if not (dest / "predictions.npz").is_file() or not (dest / "checkpoint.pt").is_file():
                    raise RuntimeError(f"Ablation failed to produce predictions and a checkpoint for {dest}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "_evaluate":
        evaluate(Path(sys.argv[2]))
    else:
        main()
