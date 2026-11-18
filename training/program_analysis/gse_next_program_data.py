"""Train-only NMF gene programs for GSE176078 cell hyperedges.

No cell labels enter gene selection, NMF fitting, or activation thresholds.
The original raw count matrix is aligned to the same 3000 cells in every fold.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import NMF

from gse_patient_state_queue import BASE_DATA, dataset_path
from gse_shared_eval import sha256, split_sha


ROOT = Path(__file__).resolve().parent.parent
CV_DATA = ROOT / "bio_runs/gse176078_cv5_20261012/data"
OUTPUT = ROOT / "bio_runs/gse176078_next_hyper_20261013/program_data_v2"


def log_cpm(counts):
    totals = np.asarray(counts.sum(axis=1)).ravel()
    normalized = sp.diags(1e4 / np.maximum(totals, 1)) @ counts
    normalized = normalized.tocsr().astype(np.float32)
    normalized.data = np.log1p(normalized.data)
    return normalized


def run(args):
    source_info = json.loads((args.cv_data / "folds.json").read_text())
    atlas = Path(source_info["source"])
    counts_path = atlas.with_name("gse176078_brca_counts.npz")
    genes_path = atlas.with_name("gse176078_brca_genes.npy")
    genes = np.load(genes_path, allow_pickle=False).astype(str)
    counts = sp.load_npz(counts_path).tocsr().astype(np.float32)
    with np.load(atlas, allow_pickle=False) as raw:
        barcodes = raw["barcodes"].copy()
    if counts.shape != (len(barcodes), len(genes)):
        raise ValueError("Atlas counts and gene names misalign")
    matrix = log_cpm(counts)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for fold in range(1, 6):
        npz = dataset_path(args.base_data, fold)
        with np.load(npz, allow_pickle=False) as data:
            train = data["train_idx"].copy()
            if not np.array_equal(data["barcodes"], barcodes):
                raise ValueError(f"Atlas cell order changed in fold {fold}")
            split = split_sha(data)
        training = matrix[train]
        mean = np.asarray(training.mean(axis=0)).ravel()
        second = np.asarray(training.multiply(training).mean(axis=0)).ravel()
        variance = np.maximum(second - mean * mean, 0)
        # Technical abundance programs should not dominate the state groups.
        eligible = np.asarray([
            not (gene.upper().startswith(("MT-", "RPL", "RPS")) or
                 gene.upper() == "MALAT1") for gene in genes])
        candidates = np.flatnonzero(eligible)
        if len(candidates) < args.genes:
            raise ValueError("Too few non-mitochondrial, non-ribosomal genes")
        selected = candidates[np.argsort(variance[candidates])[-args.genes:]]
        subset = matrix[:, selected]
        reducer = NMF(n_components=args.programs, init="nndsvda",
                      random_state=168023 + fold, max_iter=300,
                      tol=1e-4)
        reducer.fit(subset[train])
        scores = reducer.transform(subset).astype(np.float32)
        thresholds = np.quantile(scores[train], 0.75, axis=0)
        scaled = scores / np.maximum(scores[train].mean(axis=0), 1e-6)
        dominant = scaled.argmax(axis=1)
        active = np.zeros(scores.shape, dtype=np.uint8)
        rows = np.arange(len(scores))
        active[rows, dominant] = (scores[rows, dominant] >
                                  thresholds[dominant]).astype(np.uint8)
        top_gene_indices = selected[np.argsort(reducer.components_, axis=1)[:, -15:][:, ::-1]]
        target = args.output / f"fold_{fold:02d}/programs.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            with np.load(target, allow_pickle=False) as old:
                if (old["split_sha256"].item() != split or
                        not np.array_equal(old["active"], active)):
                    raise ValueError(f"Existing gene programs changed: {target}")
        else:
            np.savez_compressed(target, active=active, scores=scores,
                                thresholds=thresholds.astype(np.float32),
                                selected_gene_indices=selected.astype(np.int32),
                                top_genes=genes[top_gene_indices],
                                split_sha256=np.asarray(split),
                                barcodes=barcodes)
        record = {"fold": fold, "program_sha256": sha256(target),
                  "split_sha256": split,
                  "active_cells": int(active.any(axis=1).sum()),
                  "program_support": active.sum(axis=0).astype(int).tolist(),
                  "top_genes": genes[top_gene_indices].tolist()}
        records.append(record)
        print(json.dumps(record), flush=True)
    (args.output / "manifest.json").write_text(json.dumps({
        "source_counts_sha256": sha256(counts_path),
        "source_genes_sha256": sha256(genes_path),
        "preparation_sha256": sha256(Path(__file__)),
        "programs": args.programs, "hvg_count": args.genes,
        "fit_scope": "training patients only per fold",
        "gene_filter": "exclude MT-, RPL, RPS, MALAT1 before train-only HVG selection",
        "activation": "dominant NMF program above its training 75th percentile",
        "uses_labels": False, "folds": records}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-data", type=Path, default=BASE_DATA)
    p.add_argument("--cv-data", type=Path, default=CV_DATA)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--programs", type=int, default=12)
    p.add_argument("--genes", type=int, default=2000)
    run(p.parse_args())
