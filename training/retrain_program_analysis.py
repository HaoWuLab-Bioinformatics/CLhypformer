"""Run the source and external program analyses using newly trained checkpoints."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ATLAS = Path("bio_runs/gse176078_full_20261017")
NEURAL = ("mlp", "gcn", "gat", "hgnn", "fhnn", "lighthgnn")
SEEDS = (168023000, 168023001)
FROZEN = (
    ATLAS / "data/fold_01/biomed/preprocessing.npz",
    ATLAS / "data/folds.json",
    ATLAS / "state/MSigDB_Hallmark_2020.gmt",
    Path("data/biomed/Wu_etal_2021_BRCA_scRNASeq/metadata.csv"),
    Path("bio_runs/gse176078_caf_pca_compare_20261018/pca/fold_01.npz"),
    Path("bio_runs/gse176078_caf_mechanism_20261017/external/source"),
    Path("bio_runs/gse176078_caf_final_compare_20261018/external_inputs"),
)


def link(source, target):
    source = source.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=source.is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "program-inputs", "expression", "strict", "training", "baselines"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    cfg = parser.parse_args()
    project = cfg.project.resolve()
    training = cfg.training.resolve()
    root = training / "figure6"
    medium = root / "medium"
    medium.mkdir(parents=True)
    source = HERE.parent / "model_source/medium"
    shutil.copytree(source, medium, dirs_exist_ok=True)
    for entry in (HERE / "program_analysis").glob("*.py"):
        if not (medium / entry.name).exists():
            shutil.copy2(entry, medium)
    shutil.copy2(HERE / "gse_caf_confirm_worker_20261022.py", medium)
    for path in FROZEN:
        link(cfg.program_inputs / path, root / path)
    program = ATLAS.parent / "gse176078_patient_programs_20261022"
    for name in ("B_panels.json", "shared_program_dictionary.npz", "gene_preprocessing.npz", "retrieval_split.npz"):
        path = program / name
        link(cfg.program_inputs / path, root / path)
    for file in ("gse176078_brca_counts.npz", "gse176078_brca_genes.npy"):
        link(cfg.expression / file, root / ATLAS / "data" / file)
    for path in ("data/fold_01/biomed/gse176078_brca.npz", "state/fold_01/biostate.npz"):
        link(project / ATLAS / path, root / ATLAS / path)
    runs = root / ATLAS / "runs"
    runs.mkdir()
    digests = {name: hashlib.sha256((medium / name).read_bytes()).hexdigest()
               for name in ("hypformer.py", "gse_biostate_model.py")}
    (runs / "plan.json").write_text(json.dumps({"code_sha256": digests}), encoding="utf-8")
    for seed in SEEDS:
        fold = Path(f"fold_01/seed_{seed}")
        link(training / "figure5" / fold, runs / "state_hyper_cl" / fold)
        for method in NEURAL:
            dest = root / ATLAS.parent / "gse176078_caf_confirm_20261022/selected" / method / fold
            link(training / "figure5_baselines" / method / fold / "selected", dest)
    revision = root / "revision_runs/single_cell"
    link(cfg.strict, revision / "strict")
    for protocol in ("standard", "strict"):
        link(training / "cell_ablations" / protocol, revision / "paired" / protocol)
    env = dict(os.environ, CLHYP_ATLAS_ROOT=str(root), CLHYP_BASELINES_ROOT=str(cfg.baselines.resolve()))
    analyses = (
        ["-c", "from export_program_coordinates import source_usage; source_usage()"],
        [str(medium / "gse_program_classifiers_20261022.py")],
        [str(medium / "gse_program_confirm_20261023.py")],
        [str(medium / "prepare_program_figure_data_20261024.py")],
        ["-c", "from export_program_coordinates import run; run()"],
        ["-c", "from yunxing_revision_single_cell import retrieve; retrieve('standard'); retrieve('strict')"],
    )
    for args in analyses:
        subprocess.run([sys.executable, *args], cwd=medium, env=env, check=True)


if __name__ == "__main__":
    main()
