"""Frozen paired ablations of patient-local task edges and cell-state hyperedges.

Stage 1 compares original HypFormer, existing fusion, local task graph only,
local hyperedges only, and both structure changes with CL weight zero.
Stage 2 adds the leave-one-patient-out prototype loss only to the same
two-module structure. The primary task is inductive new-patient prediction;
transductive results are recorded separately. No test score selects a stage.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from gse_shared_eval import ORIGINAL_MEDIUM, sha256, split_sha


MEDIUM = Path(__file__).resolve().parent
ROOT = MEDIUM.parent
BASE_DATA = ROOT / "bio_runs/gse176078_rare_diagnostic_v2_20261012/balanced_data"
LOCAL_DATA = ROOT / "bio_runs/gse176078_patient_state_v2_20261013/data"
OUTPUT = ROOT / "bio_runs/gse176078_patient_state_factorial_20261013"
SEEDS = tuple(range(168023000, 168023010))
PROTOCOLS = ("inductive", "transductive")
STAGE_1 = ("original", "fusion_legacy", "graph_local", "hyper_local",
           "both_local")
STAGE_2 = ("both_local_prototype",)
FUSION = {
    "model": "modified", "use_cl": 1,
    "cl_mode": "patient_prototype", "prototype_temperature": 0.2,
    "fine_labels": 0, "cross_patient": 1,
    "no_pretrain": True, "aux_head": 0, "consistency": 0,
    "cl_warmup_epochs": 10, "cl_ramp_epochs": 20,
    "min_checkpoint_epoch": 30,
}
SCENARIOS = {
    "original": {"model": "original", "use_cl": 0,
                 "no_pretrain": True, "min_checkpoint_epoch": 30,
                 "graph_policy": "global_knn"},
    "fusion_legacy": {**FUSION, "cl_weight": 0.0,
                      "graph_policy": "global_knn",
                      "hyperedge_policy": "global_learned"},
    "graph_local": {**FUSION, "cl_weight": 0.0,
                    "graph_policy": "patient_local",
                    "hyperedge_policy": "global_learned"},
    "hyper_local": {**FUSION, "cl_weight": 0.0,
                    "graph_policy": "global_knn",
                    "hyperedge_policy": "patient_state"},
    "both_local": {**FUSION, "cl_weight": 0.0,
                   "graph_policy": "patient_local",
                   "hyperedge_policy": "patient_state"},
    "both_local_prototype": {**FUSION, "cl_weight": 0.1,
                             "graph_policy": "patient_local",
                             "hyperedge_policy": "patient_state"},
}
PAIRS = (("fusion_legacy", "original"),
         ("graph_local", "fusion_legacy"),
         ("hyper_local", "fusion_legacy"),
         ("both_local", "graph_local"),
         ("both_local", "hyper_local"),
         ("both_local", "fusion_legacy"),
         ("both_local_prototype", "both_local"),
         ("both_local", "original"))


def event(root, **fields):
    item = {"time": datetime.now(timezone.utc).isoformat(), **fields}
    with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item) + "\n")
    print(json.dumps(item), flush=True)


def code_hashes():
    paths = {
        "original_model": ORIGINAL_MEDIUM / "hypformer.py",
        "modified_model": MEDIUM / "hypformer.py",
        "patient_state_graph": MEDIUM / "gse_patient_state_graph.py",
        "patient_local_data": MEDIUM / "gse_patient_local_data.py",
        "patient_modules": MEDIUM / "gse_patient_modules.py",
        "main": MEDIUM / "main.py", "dataset": MEDIUM / "dataset.py",
        "shared_evaluator": MEDIUM / "gse_shared_eval.py",
        "inductive_evaluator": MEDIUM / "gse_inductive_eval.py",
        "patient_analysis": MEDIUM / "gse_patient_state_analysis.py",
        "analysis_core": MEDIUM / "gse_confirmation_analysis.py",
        "queue": Path(__file__).resolve(),
    }
    return {name: sha256(path) for name, path in paths.items()}


def flags(options):
    result = []
    for name, value in options.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(flag)
        else:
            result.extend((flag, str(value)))
    return result


def dataset_path(root, fold):
    return root / f"fold_{fold:02d}/biomed/gse176078_brca.npz"


def prepare_manifest(args):
    graph_manifest = args.local_data / "graph_manifest.json"
    if not graph_manifest.is_file():
        raise FileNotFoundError(graph_manifest)
    data_hashes, split_hashes = {}, {}
    for name, root in (("global_knn", args.base_data),
                       ("patient_local", args.local_data)):
        data_hashes[name], split_hashes[name] = {}, {}
        for fold in range(1, 6):
            file = dataset_path(root, fold)
            if not file.is_file():
                raise FileNotFoundError(file)
            data_hashes[name][str(fold)] = sha256(file)
            with np.load(file, allow_pickle=False) as data:
                split_hashes[name][str(fold)] = split_sha(data)
    if split_hashes["global_knn"] != split_hashes["patient_local"]:
        raise ValueError("Local graph changed a patient split")
    manifest = {
        "source_data": str(args.base_data), "local_data": str(args.local_data),
        "local_graph_manifest_sha256": sha256(graph_manifest),
        "output": str(args.output), "data_sha256": data_hashes,
        "split_sha256": split_hashes["global_knn"],
        "code_sha256": code_hashes(), "seeds": SEEDS,
        "folds": list(range(1, 6)), "protocols": PROTOCOLS,
        "stage_1": STAGE_1, "stage_2": STAGE_2,
        "scenarios": SCENARIOS, "predefined_pairs": PAIRS,
        "epochs": args.epochs, "patience": args.patience,
        "checkpoint_metric": "macro_f1", "min_checkpoint_epoch": 30,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": args.device, "analysis_unit": "patient",
        "test_metrics_used_for_design_selection": False,
        "cohort_caveat": "Same 26 patients previously examined; exploratory",
    }
    path = args.output / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != json.loads(json.dumps(manifest)):
            raise ValueError("Cannot resume with changed code, data, or settings")
    else:
        path.write_text(json.dumps(manifest, indent=2))
    return manifest


def run_case(args, manifest, protocol, scenario, fold, seed):
    options = SCENARIOS[scenario]
    policy = options["graph_policy"]
    npz = dataset_path(args.base_data if protocol == "inductive" or
                       policy == "global_knn" else args.local_data, fold)
    case = args.output / protocol / scenario / f"fold_{fold:02d}" / f"seed_{seed}"
    case.mkdir(parents=True, exist_ok=True)
    metrics_file = case / "metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
        if (metrics["seed"] != seed or metrics["data_sha256"] != sha256(npz) or
                metrics["split_sha256"] != manifest["split_sha256"][str(fold)] or
                metrics["checkpoint_metric"] != "macro_f1" or
                metrics["min_checkpoint_epoch"] != 30):
            raise ValueError(f"Completed case mismatches manifest: {case}")
        return
    if code_hashes() != manifest["code_sha256"]:
        raise RuntimeError("Frozen source changed during the experiment")
    if (os.environ.get("CUDA_VISIBLE_DEVICES") !=
            manifest["cuda_visible_devices"]):
        raise RuntimeError("GPU mapping changed during the experiment")
    if (sha256(args.local_data / "graph_manifest.json") !=
            manifest["local_graph_manifest_sha256"]):
        raise RuntimeError("Patient-local graph manifest changed")
    expected = manifest["data_sha256"]["global_knn" if protocol ==
        "inductive" else policy][str(fold)]
    if sha256(npz) != expected:
        raise RuntimeError(f"Fold data changed: {npz}")
    evaluator = "gse_inductive_eval.py" if protocol == "inductive" else "gse_shared_eval.py"
    options_for_cli = {name: value for name, value in options.items()
                       if name != "graph_policy" or protocol == "inductive"}
    command = [sys.executable, str(MEDIUM / evaluator),
               "--npz", str(npz), "--output-dir", str(case),
               "--seed", str(seed), "--device", str(args.device),
               "--epochs", str(args.epochs), "--patience", str(args.patience),
               "--checkpoint-metric", "macro_f1", *flags(options_for_cli)]
    event(args.output, status="started", protocol=protocol, scenario=scenario,
          fold=fold, seed=seed)
    with (case / "worker.log").open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, cwd=MEDIUM, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONHASHSEED": str(seed),
                 "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"})
    if completed.returncode != 0 or not metrics_file.is_file():
        event(args.output, status="failed", protocol=protocol, scenario=scenario,
              fold=fold, seed=seed, exit_code=completed.returncode)
        raise RuntimeError(f"Training failed: {case}")
    event(args.output, status="done", protocol=protocol, scenario=scenario,
          fold=fold, seed=seed)


def run_stage(args, manifest, stage, scenarios):
    event(args.output, status="stage_started", stage=stage,
          scenarios=list(scenarios))
    for protocol in PROTOCOLS:
        for fold in range(1, 6):
            for seed in SEEDS:
                for scenario in scenarios:
                    run_case(args, manifest, protocol, scenario, fold, seed)
    event(args.output, status="stage_finished", stage=stage)


def summarize(args):
    result = {"manifest_sha256": sha256(args.output / "manifest.json"),
              "protocols": {}}
    for protocol in PROTOCOLS:
        data = {}
        section = {"scenarios": {}, "paired": {}}
        for scenario in SCENARIOS:
            cases = {}
            for fold in range(1, 6):
                for seed in SEEDS:
                    file = (args.output / protocol / scenario / f"fold_{fold:02d}" /
                            f"seed_{seed}" / "metrics.json")
                    cases[(fold, seed)] = json.loads(file.read_text())
            data[scenario] = cases
            section["scenarios"][scenario] = {
                "mean_fold_seed_accuracy": float(np.mean(
                    [x["test_accuracy"] for x in cases.values()])),
                "mean_fold_seed_macro_f1": float(np.mean(
                    [x["test_macro_f1"] for x in cases.values()])),
                "mean_fold_seed_plasmablast_recall": float(np.mean(
                    [x["class_recall"][7] for x in cases.values()])),
            }
        for left, right in PAIRS:
            difference = np.asarray([data[left][key]["test_accuracy"] -
                                     data[right][key]["test_accuracy"]
                                     for key in sorted(data[left])])
            section["paired"][f"{left}_vs_{right}"] = {
                "mean_accuracy_delta_pp": float(difference.mean()),
                "wins_ties_losses": [int((difference > 0).sum()),
                                     int((difference == 0).sum()),
                                     int((difference < 0).sum())],
                "delta_by_fold": {str(fold): float(difference[(fold - 1) * 10:
                                                fold * 10].mean())
                                  for fold in range(1, 6)},
            }
        result["protocols"][protocol] = section
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = prepare_manifest(args)
    try:
        run_stage(args, manifest, "structure_without_contrast", STAGE_1)
        run_stage(args, manifest, "loss_on_fixed_structure", STAGE_2)
        summarize(args)
        with (args.output / "analysis.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, str(MEDIUM / "gse_patient_state_analysis.py"),
                 "--data-root", str(args.base_data), "--queue-root", str(args.output),
                 "--output-dir", str(args.output / "patient_analysis")],
                cwd=MEDIUM, stdout=log, stderr=subprocess.STDOUT, check=True)
    except Exception as exc:
        event(args.output, status="finished_with_errors", error=str(exc))
        raise
    event(args.output, status="finished", failures=0,
          completed=len(SCENARIOS) * len(PROTOCOLS) * 5 * len(SEEDS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, default=BASE_DATA)
    parser.add_argument("--local-data", type=Path, default=LOCAL_DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    args.base_data = args.base_data.resolve()
    args.local_data = args.local_data.resolve()
    args.output = args.output.resolve()
    if args.background:
        args.output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--base-data", str(args.base_data),
                   "--local-data", str(args.local_data),
                   "--output", str(args.output),
                   "--device", str(args.device),
                   "--epochs", str(args.epochs),
                   "--patience", str(args.patience)]
        with (args.output / "queue.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=MEDIUM, stdout=log,
                                       stderr=subprocess.STDOUT,
                                       start_new_session=True)
        (args.output / "queue.pid").write_text(str(process.pid))
        print(f"Patient-state queue PID={process.pid} output={args.output}")
    else:
        run(args)


if __name__ == "__main__":
    main()
