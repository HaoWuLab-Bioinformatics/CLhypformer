"""Paired full-atlas revision experiments using the frozen patient split."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

MEDIUM = Path(__file__).resolve().parent
BASE = MEDIUM.parent / "bio_runs/gse176078_full_20261017"
ROOT = MEDIUM.parent / "revision_runs/single_cell"
PROGRAM = BASE.parent / "gse176078_patient_programs_20261022"
SEEDS = (168023000, 168023001)
ARMS = {
    "local": ("removed", 0., False),
    "fixed_fusion": ("nonlinear", 0., True),
    "gated_fusion": ("nonlinear", 0., False),
    "full": ("nonlinear", .001, False),
}


def prepare():
    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    from gse_next_program_data import log_cpm
    from gse_patient_state_graph import patient_local_edges
    from gse_biostate_data import prepare as hallmark

    root = ROOT / "strict"
    state = root / "state/fold_01/biostate.npz"
    if state.exists():
        return
    data = root / "data"
    folder = data / "fold_01/biomed"
    folder.mkdir(parents=True, exist_ok=True)
    banned = {g.upper() for panel in json.loads((PROGRAM / "B_panels.json").read_text()) for g in panel["genes"]}
    genes = np.load(BASE / "data/gse176078_brca_genes.npy").astype(str)
    keep = ~np.isin(np.char.upper(genes), sorted(banned))
    assert len(set(np.char.upper(genes[~keep]))) == len(banned)
    counts = sp.load_npz(BASE / "data/gse176078_brca_counts.npz")[:, keep].tocsr()
    atlas = data / "gse176078_brca.npz"
    shutil.copyfile(BASE / "data/gse176078_brca.npz", atlas)
    np.save(atlas.with_name("gse176078_brca_genes.npy"), genes[keep])
    sp.save_npz(atlas.with_name("gse176078_brca_counts.npz"), counts)
    (data / "folds.json").write_text(json.dumps(dict(source=str(atlas), excluded_genes=sorted(banned))))
    normalized = log_cpm(counts)
    del counts
    with np.load(BASE / "data/fold_01/biomed/gse176078_brca.npz") as d:
        arrays = {k: d[k] for k in d.files if k not in ("features", "edge_index")}
    train = arrays["train_idx"]
    reducer = TruncatedSVD(50, random_state=168018799).fit(normalized[train])
    features = reducer.transform(normalized)
    del normalized
    scaler = StandardScaler().fit(features[train])
    arrays["features"] = scaler.transform(features).astype(np.float32)
    arrays["edge_index"], graph = patient_local_edges(arrays["features"], arrays["patient_ids"], k=5)
    np.savez_compressed(folder / "gse176078_brca.npz", **arrays)
    np.savez_compressed(folder / "preprocessing.npz", components=reducer.components_,
                        mean=scaler.mean_, scale=scaler.scale_)
    state.parent.parent.mkdir(exist_ok=True)
    shutil.copyfile(BASE / "state/MSigDB_Hallmark_2020.gmt", state.parent.parent / "MSigDB_Hallmark_2020.gmt")
    hallmark(SimpleNamespace(output=state.parent.parent, cv_data=data, base_data=data, folds=(1,)))
    with np.load(state) as z:
        assert not banned.intersection(np.char.upper(z["selected_gene_names"].astype(str)))
    print("STRICT_INPUT_READY", len(banned), graph, flush=True)


def evaluate():
    if os.environ.get("REVISION_FIXED_GATE") == "1":
        from gse_biostate_model import BiologicalStateHypergraph
        original = BiologicalStateHypergraph.__init__

        def initialize(self, *args, **kwargs):
            original(self, *args, **kwargs)
            self.gate.requires_grad_(False)
            self.state_gate.requires_grad_(False)

        BiologicalStateHypergraph.__init__ = initialize
    from gse176078_full_eval import main
    sys.argv.pop(1)
    main()


def run_case(protocol, arm, seed, smoke):
    mode, weight, fixed = ARMS[arm]
    source = BASE if protocol == "standard" else ROOT / "strict"
    original = BASE / f"runs/state_hyper_cl/fold_01/seed_{seed}/command.json"
    command = json.loads(original.read_text())
    command[:2] = [sys.executable, "-u", __file__, "evaluate"]
    case = ROOT / ("smoke" if smoke else "paired") / protocol / arm / str(seed)
    case.mkdir(parents=True, exist_ok=True)
    if (case / "complete").exists():
        return
    options = {
        "--state-mode": mode, "--cl-weight": str(weight),
        "--state-data": str(source / "state/fold_01/biostate.npz"),
        "--npz": str(source / "data/fold_01/biomed/gse176078_brca.npz"),
        "--output-dir": str(case),
    }
    if smoke:
        options.update({"--epochs": "2", "--min-checkpoint-epoch": "1"})
    for k, v in options.items():
        command[command.index(k) + 1] = v
    (case / "command.json").write_text(json.dumps(command, indent=2))
    print("START", protocol, arm, seed, flush=True)
    with (case / "train.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, "REVISION_FIXED_GATE": str(int(fixed)), "PYTHONHASHSEED": str(seed)})
    (case / "complete").touch()
    print("DONE", protocol, arm, seed, flush=True)


def retrieve(protocol):
    import pandas as pd
    import scipy.sparse as sp
    import torch
    from sklearn.neighbors import NearestNeighbors
    import gse_caf_mechanism_runtime_20261017 as runtime
    from gse_biostate_model import BiologicalStateHypergraph
    from gse_caf_final_compare_20261018 import final_embedding
    import gse_program_confirm_20261023 as targets
    from gse_patient_programs_20261022 import fit_coordinates, project, residual, rho

    torch.set_num_threads(3)
    _, source_targets = targets.prepare_source()
    with np.load(PROGRAM / "retrieval_split.npz") as z:
        ref = z["reference_local"]
    model_data = BASE if protocol == "standard" else ROOT / "strict"
    folder = ROOT / "paired" / protocol
    with np.load(model_data / "data/fold_01/biomed/gse176078_brca.npz") as z:
        x, groups, edges = z["features"], z["patient_ids"], z["edge_index"]
    with np.load(model_data / "state/fold_01/biostate.npz") as z:
        prior = z["prior"]
    excluded = set(json.loads((model_data / "data/folds.json").read_text()).get("excluded_genes", []))
    source_idx = source_targets["idx"]

    def representations(model, x, prior, groups, edges, wanted):
        points = np.empty((len(wanted), 512), dtype=np.float32)
        for group in np.unique(groups[wanted]):
            ids = np.flatnonzero(groups == group)
            where = np.flatnonzero(groups[wanted] == group)
            mapping = np.full(len(groups), -1, dtype=np.int64)
            mapping[ids] = np.arange(len(ids))
            keep = (mapping[edges[0]] >= 0) & (mapping[edges[1]] >= 0)
            z, _, _ = final_embedding(model, x[ids], prior[ids], np.zeros(len(ids), np.int64), mapping[edges[:, keep]])
            points[where] = z[mapping[wanted[where]]]
        return points

    from gse_patient_state_graph import patient_local_edges
    rows = []
    for external in targets.object_folders():
        print("RETRIEVAL", protocol, external.name, flush=True)
        ids, _, patients, scores, _, depth, _ = targets.external_targets(external, source_targets)
        meta = pd.read_csv(external / "metadata.csv", dtype=str)
        counts = sp.load_npz(external / "counts.npz").tocsr()
        genes = np.load(external / "genes.npy").astype(str)
        keep = ~np.isin(np.char.upper(genes), sorted(excluded))
        runtime.BASE = model_data
        ex, ep, _ = runtime.project_external(counts[:, keep], genes[keep], 1)
        runtime.BASE = BASE
        sample = meta["sample"].to_numpy(str)
        ee, _ = patient_local_edges(ex, sample, k=5)
        for seed in SEEDS:
            for arm in (*ARMS, "pca"):
                if protocol == "strict" and arm in ("fixed_fusion", "gated_fusion"):
                    continue
                if arm == "pca":
                    sx, qx = x[source_idx], ex[ids]
                else:
                    checkpoint = torch.load(folder / arm / str(seed) / "checkpoint.pt", map_location="cpu",
                                            weights_only=False)["state_dict"]
                    model = runtime.load_full(1, seed)
                    model.state_module = BiologicalStateHypergraph(
                        50, len(checkpoint["state_module.anchors"]), model.manifold_hidden, mode=ARMS[arm][0])
                    for name in ("state_positive", "state_negative"):
                        setattr(model, name, checkpoint[name].clone())
                    model.load_state_dict(checkpoint, strict=True)
                    model.eval()
                    sx = representations(model, x, prior, groups, edges, source_idx)
                    qx = representations(model, ex, ep, sample, ee, ids)
                train = source_targets["role"] == 0
                params = fit_coordinates(sx[train], source_targets["patients"][train], 15)
                neighbors = NearestNeighbors(n_neighbors=10).fit(project(sx[ref], params))
                local = neighbors.kneighbors(project(qx, params), return_distance=False)
                prediction = source_targets["scores"][ref[local]].mean(1)
                for patient in np.unique(patients):
                    take = patients == patient
                    if take.sum() < 30:
                        continue
                    for p in (0, 1):
                        value = rho(residual(prediction[take, p], depth[take]), residual(scores[take, p], depth[take]))
                        rows.append(dict(object=external.name, patient=patient, arm=arm, seed=seed,
                                         program=p, rho_depth=value, cells=int(take.sum())))
        pd.DataFrame(rows).to_csv(folder / "external_patient_metrics.partial.csv", index=False)
    result = pd.DataFrame(rows)
    duplicates = result.groupby("patient")["object"].nunique()
    result["eligible_primary"] = ~result["patient"].isin(duplicates[duplicates > 1].index)
    result.to_csv(folder / "external_patient_metrics.csv", index=False)
    result[result["eligible_primary"]].groupby(["arm", "program"])["rho_depth"].mean().to_csv(folder / "external_mean.csv")


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
    elif sys.argv[1:2] == ["evaluate"]:
        evaluate()
    elif sys.argv[1:] == ["--smoke"]:
        run_case("standard", "fixed_fusion", SEEDS[0], True)
    else:
        prepare()
        for protocol, arms in (("standard", ARMS), ("strict", ("local", "full"))):
            for seed in SEEDS:
                for arm in arms:
                    run_case(protocol, arm, seed, False)
        for protocol in ("strict", "standard"):
            retrieve(protocol)

