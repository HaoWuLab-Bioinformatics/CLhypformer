"""Training-only Hallmark scoring and cross-patient program matching.

This is a standardized-expression score, not GSEA enrichment. No cell-type,
minor-state, subtype or held-out label enters this preparation.
"""
import argparse
import json
from pathlib import Path
import urllib.request

import numpy as np
import scipy.sparse as sp

from gse_next_program_data import CV_DATA, log_cpm
from gse_patient_state_queue import BASE_DATA, dataset_path
from gse_shared_eval import sha256, split_sha

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'bio_runs/gse176078_biostate_20261016/data'
URL = 'https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020'


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    gmt = args.output / 'MSigDB_Hallmark_2020.gmt'
    if not gmt.exists():
        request = urllib.request.Request(URL, headers={'User-Agent': 'GSE176078-research/1.0'})
        with urllib.request.urlopen(request, timeout=45) as response:
            content = response.read()
        text = content.decode('utf-8')
        if len(text.splitlines()) != 50 or '<html' in text.lower():
            raise ValueError('Expected a 50-program Enrichr Hallmark GMT')
        gmt.write_bytes(content)
    programs = {}
    for line in gmt.read_text().splitlines():
        fields = line.strip().split('\t')
        programs[fields[0]] = set(g.split(',')[0].upper() for g in fields[2:] if g)
    info = json.loads((args.cv_data / 'folds.json').read_text())
    atlas = Path(info['source'])
    genes = np.load(atlas.with_name('gse176078_brca_genes.npy'), allow_pickle=False).astype(str)
    counts_path = atlas.with_name('gse176078_brca_counts.npz')
    counts = sp.load_npz(counts_path).tocsr()
    with np.load(atlas, allow_pickle=False) as source:
        barcodes = source['barcodes'].copy()
    if counts.shape != (len(barcodes), len(genes)):
        raise ValueError('Counts, cells and genes are misaligned')
    names = sorted(programs)
    gene_upper = np.char.upper(genes)
    masks = np.stack([np.isin(gene_upper, list(programs[n])) for n in names])
    coverage = masks.sum(1)
    if int(coverage.min()) < 10:
        raise ValueError('Insufficient gene coverage for a Hallmark program')
    selected = np.flatnonzero(masks.any(0))
    expression = log_cpm(counts)[:, selected].toarray().astype(np.float32)
    weights = masks[:, selected].astype(np.float32) / coverage[:, None]
    gene_jaccard = np.empty((len(names), len(names)), dtype=np.float32)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            gene_jaccard[i, j] = len(programs[a] & programs[b]) / max(1, len(programs[a] | programs[b]))
    records = []
    for fold in getattr(args, 'folds', (1, 3, 5)):
        with np.load(dataset_path(args.base_data, fold), allow_pickle=False) as data:
            train = data['train_idx'].copy()
            features = data['features'][train].copy()
            patients = data['patient_ids'][train].astype(str)
            if not np.array_equal(data['barcodes'], barcodes):
                raise ValueError('Fold and raw count barcodes are not aligned')
            split = split_sha(data)
        mean, sd = expression[train].mean(0), np.maximum(expression[train].std(0), 0.1)
        gene_z = np.clip((expression - mean) / sd, -5, 5)
        score = gene_z @ weights.T
        p_mean, p_sd = score[train].mean(0), np.maximum(score[train].std(0), 0.05)
        score = (score - p_mean) / p_sd
        # Positive excess above a fixed training-standardized threshold. A cell
        # with no active program retains its local encoder representation.
        prior = np.maximum(score - 0.5, 0).astype(np.float32)
        order = np.argsort(prior, axis=1)[:, -3:]
        sparse = np.zeros_like(prior)
        np.put_along_axis(sparse, order, np.take_along_axis(prior, order, axis=1), axis=1)
        sparse /= np.maximum(sparse.sum(1, keepdims=True), 1e-8)
        names_patient, codes = np.unique(patients, return_inverse=True)
        m, p = len(names), len(names_patient)
        contexts = np.zeros((p, m, features.shape[1]), dtype=np.float32)
        supports = np.zeros((p, m), dtype=np.int64)
        for group in range(p):
            h = sparse[train][codes == group]
            contexts[group] = (h.T @ features[codes == group]) / np.maximum(h.sum(0)[:, None], 1e-8)
            supports[group] = (h > 0).sum(0)
        unit = contexts / np.maximum(np.linalg.norm(contexts, axis=2, keepdims=True), 1e-8)
        positive = np.zeros((p * m, p * m), dtype=bool)
        for program in range(m):
            similarities = unit[:, program] @ unit[:, program].T
            valid = (supports[:, program] >= 3)
            allowed = valid[:, None] & valid[None, :] & ~np.eye(p, dtype=bool)
            similarities = np.where(allowed & (similarities >= 0.5), similarities, -np.inf)
            near = np.zeros((p, p), dtype=bool)
            for group in range(p):
                ids = np.argsort(similarities[group])[-min(2, p):]
                near[group, ids[np.isfinite(similarities[group, ids])]] = True
            a, b = np.where(near & near.T)
            positive[a * m + program, b * m + program] = True
        corr = np.nan_to_num(np.corrcoef(score[train].T), nan=1.0)
        negative_program = (gene_jaccard < 0.1) & (corr < 0.5) & ~np.eye(m, dtype=bool)
        target = args.output / f'fold_{fold:02d}/biostate.npz'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            with np.load(target, allow_pickle=False) as old:
                if old['split_sha256'].item() != split or not np.array_equal(old['prior'], sparse):
                    raise ValueError('Existing frozen biological program data differ')
        else:
            np.savez_compressed(target, prior=sparse, scores=score.astype(np.float32),
                                program_names=np.asarray(names), train_idx=train,
                                train_patient_names=names_patient,
                                train_patient_codes=codes, train_edge_positive=positive,
                                negative_program=negative_program,
                                selected_gene_indices=selected, selected_gene_names=genes[selected],
                                gene_training_mean=mean, gene_training_sd=sd,
                                program_training_mean=p_mean, program_training_sd=p_sd,
                                gene_overlap=gene_jaccard, training_activity_correlation=corr,
                                program_gene_counts=coverage, barcodes=barcodes,
                                split_sha256=np.asarray(split))
        record = dict(fold=fold, split_sha256=split, sha256=sha256(target), programs=m,
                      cells_with_program=int((sparse.sum(1) > 0).sum()),
                      mean_memberships=float((sparse > 0).sum(1).mean()),
                      matched_directed_edge_pairs=int(positive.sum()),
                      gene_counts=coverage.tolist())
        if not positive.any():
            raise ValueError('No expression-compatible cross-patient state pairs')
        records.append(record)
        print(json.dumps(record), flush=True)
    manifest = dict(source_url=URL, gmt_sha256=sha256(gmt), counts_sha256=sha256(counts_path),
                    source_code_sha256=sha256(Path(__file__)), uses_labels=False,
                    score='mean train-standardized logCPM, train-standardized program score',
                    membership='positive excess above 0.5, top 3 overlapping programs',
                    pair_rule='same program, distinct training patients, support >=3, mutual top2 context cosine >=0.5',
                    files=records, biological_claim='Gene-set activity proxy, not validated cell state or enrichment significance.')
    path = args.output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Frozen manifest differs')
    path.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-data', type=Path, default=BASE_DATA)
    parser.add_argument('--cv-data', type=Path, default=CV_DATA)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    prepare(parser.parse_args())
