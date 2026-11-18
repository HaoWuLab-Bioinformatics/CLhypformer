"""All 100064 cells, frozen donor splits, train-only transforms, cached local graphs."""
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from gse_patient_state_graph import patient_local_edges
from gse_shared_eval import sha256, split_sha

MEDIUM = Path(__file__).resolve().parent
ROOT = MEDIUM.parent / 'bio_runs/gse176078_full_20261017'
SOURCE = MEDIUM.parent / 'data/biomed'
OLD = MEDIUM.parent / 'bio_runs/gse176078_rare_diagnostic_v2_20261012/balanced_data'


def progress(stage, **extra):
    value = dict(stage=stage, time_unix=time.time(), **extra)
    (ROOT / 'status.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)


def prepare():
    destination = ROOT / 'data'
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'folds.json').exists():
        manifest = json.loads((destination / 'folds.json').read_text())
        assert manifest['cells'] == 100064 and manifest['sampling'] == 'none'
        for record in manifest['folds']:
            path = destination / f"fold_{record['fold']:02d}/biomed/gse176078_brca.npz"
            assert sha256(path) == record['sha256']
        return
    progress('reading_full_counts')
    raw = SOURCE / 'Wu_etal_2021_BRCA_scRNASeq'
    metadata = pd.read_csv(raw / 'metadata.csv', low_memory=False)
    barcodes = pd.read_csv(raw / 'count_matrix_barcodes.tsv', header=None).iloc[:, 0].to_numpy(str)
    genes = pd.read_csv(raw / 'count_matrix_genes.tsv', header=None).iloc[:, 0].to_numpy(str)
    assert len(metadata) == 100064 and np.array_equal(metadata.iloc[:, 0].to_numpy(str), barcodes)
    labels = metadata['celltype_major'].to_numpy(str)
    names, codes = np.unique(labels, return_inverse=True)
    patients = metadata['orig.ident'].to_numpy(str)
    annotation = dict(labels=codes, class_names=names, barcodes=barcodes,
        patient_ids=patients, celltype_minor=metadata['celltype_minor'].to_numpy(str),
        subtypes=metadata['subtype'].to_numpy(str), source_node_idx=np.arange(len(metadata)))
    atlas = destination / 'gse176078_brca.npz'
    np.savez_compressed(atlas, **annotation)
    np.save(atlas.with_name('gse176078_brca_genes.npy'), genes, allow_pickle=False)
    cache = SOURCE / 'gse176078_counts_csc.npz'
    counts = sp.load_npz(cache).T.tocsr().astype(np.float32)
    assert counts.shape == (100064, 29733) and len(genes) == 29733
    sp.save_npz(atlas.with_name('gse176078_brca_counts.npz'), counts)
    progress('normalizing_full_counts', nonzeros=int(counts.nnz))
    normalized = sp.diags(1e4 / np.maximum(np.asarray(counts.sum(1)).ravel(), 1)) @ counts
    normalized = normalized.tocsr()
    normalized.data = np.log1p(normalized.data)
    del counts
    records = []
    for fold in range(1, 6):
        started = time.monotonic()
        with np.load(OLD / f'fold_{fold:02d}/biomed/gse176078_brca.npz') as old:
            groups = {key: np.unique(old['patient_ids'][old[key]]) for key in ('train_idx', 'valid_idx', 'test_idx')}
        indices = {key: np.flatnonzero(np.isin(patients, group)) for key, group in groups.items()}
        assert sum(len(ids) for ids in indices.values()) == 100064
        assert len(set.union(*(set(group) for group in groups.values()))) == 26
        assert all(not set(groups[a]) & set(groups[b]) for a, b in
            (('train_idx','valid_idx'), ('train_idx','test_idx'), ('valid_idx','test_idx')))
        train = indices['train_idx']
        assert len(np.unique(codes[train])) == 9
        progress('fit_train_SVD50', fold=fold, train_cells=len(train))
        reducer = TruncatedSVD(50, random_state=168018798 + fold)
        reducer.fit(normalized[train])
        features = reducer.transform(normalized)
        scaler = StandardScaler().fit(features[train])
        features = scaler.transform(features).astype(np.float32)
        progress('patient_local_graph', fold=fold)
        edges, graph = patient_local_edges(features, patients, k=5)
        folder = destination / f'fold_{fold:02d}/biomed'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'gse176078_brca.npz'
        np.savez_compressed(path, **annotation, **indices, features=features, edge_index=edges)
        np.savez_compressed(folder / 'preprocessing.npz', components=reducer.components_, mean=scaler.mean_, scale=scaler.scale_)
        record = dict(fold=fold, sha256=sha256(path), split_sha256=split_sha(indices),
            **{key.replace('_idx','_cells'):len(ids) for key, ids in indices.items()},
            **{key.replace('_idx','_patients'):group.tolist() for key, group in groups.items()},
            class_support={key:np.bincount(codes[ids], minlength=9).tolist() for key, ids in indices.items()},
            graph=graph, seconds=time.monotonic()-started)
        records.append(record)
        print(json.dumps(record), flush=True)
    test_sets = [set(record['test_patients']) for record in records]
    assert sum(map(len,test_sets)) == len(set.union(*test_sets)) == 26
    manifest = dict(source=str(atlas), source_counts_sha256=sha256(cache),
        source_metadata_sha256=sha256(raw / 'metadata.csv'), cells=100064, patients=26, genes=29733,
        class_names=names.tolist(), class_support=np.bincount(codes).tolist(),
        sampling='none', features=50, folds=records, preprocessing='train-only logCPM/SVD50/standardization',
        graph='exact patient-internal mutual k5 with isolated nearest-neighbour fallback',
        full_test_coverage=True, source_code_sha256=sha256(Path(__file__)))
    (destination / 'folds.json').write_text(json.dumps(manifest, indent=2))


def states():
    target = ROOT / 'state'
    if (target / 'manifest.json').exists():
        assert len(json.loads((target / 'manifest.json').read_text())['files']) == 5
        return
    progress('full_Hallmark_scores')
    # Backward-compatible preparation now accepts all five requested folds.
    import gse_biostate_data
    gse_biostate_data.prepare(SimpleNamespace(base_data=ROOT / 'data', cv_data=ROOT / 'data',
        output=target, folds=tuple(range(1,6))))


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    prepare()
    states()
    progress('data_ready')
