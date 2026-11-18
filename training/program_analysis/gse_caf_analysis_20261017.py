"""Frozen full-model CAF case study. No fitting on held-out state labels.

Reuses the ten completed inductive full-cohort checkpoints. CPU only; does
not touch the running atlas experiments or their frozen sources.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import warnings

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import softmax
from scipy.stats import rankdata, hypergeom, wilcoxon
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score

MEDIUM = Path(__file__).resolve().parent
BASE = MEDIUM.parent / 'bio_runs/gse176078_full_20261017'
ROOT = MEDIUM.parent / 'bio_runs/gse176078_caf_20261017'
SEEDS = (168023000, 168023001)
SIGNATURES = {
    'ECM': ['COL1A1', 'COL1A2', 'COL3A1', 'FN1', 'POSTN', 'MMP11'],
    'inflammatory': ['CXCL12', 'C3', 'CFD', 'IL6', 'CCL2'],
    'contractile': ['ACTA2', 'TAGLN', 'MYL9', 'TPM2'],
    'interferon': ['ISG15', 'IFIT1', 'IFIT3', 'MX1', 'IFI6'],
    'macrophage_TREM2': ['TREM2'],
    'macrophage_lipid': ['TREM2', 'APOE', 'LPL', 'CTSD'],
    'CD8_cytotoxic': ['PRF1', 'GZMB', 'GZMH', 'GNLY', 'NKG7'],
    'CD8_exhaustion_related': ['PDCD1', 'LAG3', 'TIGIT', 'HAVCR2', 'TOX'],
}
CAF_SIGNATURES = ('ECM', 'inflammatory', 'contractile', 'interferon')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def note(stage, **fields):
    value = dict(stage=stage, time=datetime.now(timezone.utc).isoformat(), **fields)
    write(ROOT / 'status.json', value)
    print(json.dumps(value), flush=True)


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty(len(p))
    out[order] = np.minimum(1, np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1])
    return out


def dist(x, y, k):
    # All coordinates live on the future sheet with curvature -1/k.
    inner = -x[:, :1] @ y[:, :1].T + x[:, 1:] @ y[:, 1:].T
    return np.sqrt(k) * np.arccosh(np.maximum(-inner / k, 1 + 1e-7))


def mean_point(x, weights, k):
    mean = np.average(x, axis=0, weights=weights)
    norm = mean[0] ** 2 - (mean[1:] ** 2).sum()
    if norm <= 0:
        raise ValueError('Lorentz barycenter left the future timelike cone')
    return mean * np.sqrt(k / norm)  # extrinsic normalized mean, not Frechet mean


def cluster(x, groups, kappa, count, seed, starts=4):
    rng = np.random.default_rng(seed)
    _, code, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1. / sizes[code]
    best = None
    for _ in range(starts):
        centers = [x[rng.choice(len(x), p=weights / weights.sum())]]
        for j in range(1, count):
            near = dist(x, np.asarray(centers), kappa).min(1) ** 2 * weights
            if near.sum() == 0:
                raise ValueError('Degenerate state representation')
            centers.append(x[rng.choice(len(x), p=near / near.sum())])
        centers = np.asarray(centers)
        previous = None
        for _ in range(60):
            distances = dist(x, centers, kappa)
            labels = distances.argmin(1)
            if previous is not None and np.array_equal(labels, previous):
                break
            previous = labels.copy()
            for j in range(count):
                mask = labels == j
                centers[j] = mean_point(x[mask], weights[mask], kappa) if mask.any() else x[np.argmax(distances.min(1)*weights)]
        loss = float(np.average(dist(x, centers, kappa).min(1), weights=weights))
        if best is None or loss < best[0]:
            best = loss, centers.copy(), dist(x, centers, kappa).argmin(1)
    return best[1:]


def signatures(expression, genes):
    upper = np.char.upper(genes.astype(str))
    scores, coverage = {}, {}
    for name, members in SIGNATURES.items():
        selected = np.flatnonzero(np.isin(upper, members))
        if len(selected) < (1 if name == 'macrophage_TREM2' else 2):
            raise ValueError('Insufficient marker coverage: ' + name)
        scores[name] = np.asarray(expression[:, selected].mean(1)).ravel()
        coverage[name] = upper[selected].tolist()
    return scores, coverage


def export_case(fold, seed, atlas, metadata):
    """Recover exact upstream fused features and audit saved Lorentz cells."""
    import torch
    from manifolds.lorentz import Lorentz
    from gse_biostate_model import BiologicalStateHypergraph
    torch.set_num_threads(4)
    case = BASE / f'runs/state_hyper_cl/fold_{fold:02d}/seed_{seed}'
    target = ROOT / f'cases/fold_{fold:02d}/seed_{seed}'
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(case / 'checkpoint.pt', map_location='cpu', weights_only=False)
    plan = json.loads((BASE / 'runs/plan.json').read_text())
    expected = plan['code_sha256']['gse_biostate_model.py']
    if sha(MEDIUM / 'gse_biostate_model.py') != expected:
        raise ValueError('Saved full model source differs; use its exact snapshot')
    state = {key[len('state_module.'):]: value for key, value in checkpoint['state_dict'].items() if key.startswith('state_module.')}
    kappa = float(state['manifold.k'])
    module = BiologicalStateHypergraph(50, len(state['anchors']), Lorentz(k=kappa), mode='nonlinear')
    module.load_state_dict(state, strict=True)
    module.eval()
    with np.load(BASE / f'data/fold_{fold:02d}/biomed/gse176078_brca.npz') as d:
        features = d['features'].copy()
        train, test = d['train_idx'].copy(), d['test_idx'].copy()
        assert np.array_equal(d['barcodes'], atlas['barcodes'])
    train_set, test_set = set(train.tolist()), set(test.tolist())
    records, edge_records, exports = [], [], []
    max_drift = 0.
    for path in sorted((case / 'state_representations').glob('*.npz')):
        with np.load(path) as saved:
            idx = saved['global_idx'].copy()
            is_train = path.name == 'train.npz'
            if not is_train and not set(idx.tolist()).issubset(test_set):
                continue  # no validation-patient outcome selection
            if is_train and set(idx.tolist()) != train_set:
                raise ValueError('Training export differs from frozen split')
            if not np.array_equal(saved['barcodes'], atlas['barcodes'][idx]):
                raise ValueError('Export/barcode mismatch')
            h = saved['memberships'].copy()
            points = saved['cells_lorentz'].copy()
            prior = saved['activity_prior'].copy()
            names = saved['program_names'].astype(str)
            edge_points = saved['edges_lorentz'].copy()
            groups = atlas['patient_ids'][idx]
        for group_code, group in enumerate(np.unique(groups)):
            local = np.flatnonzero(groups == group)
            ids = idx[local]
            with torch.no_grad():
                fused = module(torch.from_numpy(features[ids]), torch.from_numpy(prior[local]),
                               torch.zeros(len(local), dtype=torch.long)).numpy().copy()
            actual = module.last['cells'].numpy()
            drift = float(np.max(np.abs(actual - points[local])))
            max_drift = max(max_drift, drift)
            if drift > 1e-4:
                raise ValueError(f'Restored state representation differs: {group}, {drift}')
            caf = atlas['class_names'][atlas['labels'][ids]] == 'CAFs'
            if caf.any():
                exports.append(dict(idx=ids[caf], points=points[local][caf], fused=fused[caf],
                    membership=h[local][caf], train=np.full(caf.sum(), is_train)))
            for j, name in enumerate(names):
                weights = h[local, j]
                edge_records.append(dict(fold=fold, seed=seed, patient=group, role='train' if is_train else 'test',
                    program=name, total_cells=len(ids), total_support=int((weights > 0).sum()),
                    caf_support=int(((weights > 0) & caf).sum()), caf_weight=float(weights[caf].sum()),
                    all_weight=float(weights.sum()), edge_lorentz_norm=float(-edge_points[group_code,j,0]**2 + np.square(edge_points[group_code,j,1:]).sum())))
    out = {key: np.concatenate([e[key] for e in exports]) for key in exports[0]}
    if not np.array_equal(np.sort(out['idx'][~out['train']]), np.sort(test[np.isin(test, out['idx'])])):
        raise ValueError('Incomplete CAF test export')
    norm = -out['points'][:, 0]**2 + np.square(out['points'][:, 1:]).sum(1)
    if not np.allclose(norm, -kappa, atol=1e-7) or not (out['points'][:,0] > 0).all():
        raise ValueError('Invalid Lorentz coordinates')
    np.savez_compressed(target / 'caf_representations.npz', **out, program_names=names, curvature_k=kappa)
    pd.DataFrame(edge_records).to_csv(target / 'patient_program_edges.csv', index=False)
    write(target / 'provenance.json', dict(checkpoint_sha256=sha(case/'checkpoint.pt'), model_source_sha256=expected,
        saved_representation_max_abs_drift=max_drift, k=kappa, curvature=-1/kappa,
        representation='cells_lorentz: actual state-branch output; fused: actual input to task backbone, not its final hidden layer',
        program_edges='Separate patient x program edges; cross-patient similarity is not a direct shared hyperedge',
        state_fit_uses_train_patients_only=True, fine_labels_used_to_fit_states=False,
        frozen_split_sha256=plan['data'][str(fold)]['split_sha256'] if 'split_sha256' in plan['data'][str(fold)] else json.loads((case/'metrics.json').read_text())['split_sha256']))
    return out, kappa, target


def balanced_sample(groups, maximum, rng):
    unique = np.unique(groups)
    selected = []
    per = max(1, maximum // len(unique))
    for group in unique:
        ids = np.flatnonzero(groups == group)
        selected.extend(rng.choice(ids, min(per, len(ids)), replace=False))
    return np.sort(selected)


def analyze_case(fold, seed, atlas, meta, marker_scores):
    note('caf_state_discovery', fold=fold, seed=seed)
    data, kappa, target = export_case(fold, seed, atlas, meta)
    idx, points, training = data['idx'], data['points'], data['train']
    groups = atlas['patient_ids'][idx]
    rng = np.random.default_rng(seed+fold)
    x, group = points[training], groups[training]
    sample = balanced_sample(group, 1200, rng)
    distances = dist(x[sample], x[sample], kappa)
    np.fill_diagonal(distances, 0)
    trials = []
    for count in range(2, 7):
        centers, labels = cluster(x, group, kappa, count, seed+100*fold+count)
        sizes = np.bincount(labels, minlength=count)
        covered = [sum(int(((labels == j) & (group == p)).sum()) >= 5 for p in np.unique(group)) for j in range(count)]
        accepted = bool((sizes >= 30).all() and min(covered) >= 3 and len(np.unique(labels[sample])) > 1)
        silhouette = float(silhouette_score(distances, labels[sample], metric='precomputed')) if accepted else None
        trials.append(dict(count=count, centers=centers, labels=labels, sizes=sizes.tolist(),
            donors_with_5_cells=covered, eligible=accepted, silhouette=silhouette))
    admissible = [t for t in trials if t['eligible']]
    write(target / 'candidate_states.json', [{k:v for k,v in t.items() if k not in ('centers','labels')} for t in trials])
    if not admissible:
        write(target / 'state_status.json', dict(stage='no_reproducible_state_partition', reason='No train-only candidate meets support rules; no outcome-driven fallback'))
        return None
    chosen = max(admissible, key=lambda t:(t['silhouette'], -t['count']))
    centers, count = chosen['centers'], chosen['count']
    labels = dist(points, centers, kappa).argmin(1)
    # Temperature is train-fitted geometric scale, not tuned on outcomes.
    temperature = max(float(np.median(dist(x, centers, kappa).min(1))), .05)
    memberships = softmax(-dist(points, centers, kappa)/temperature, axis=1)
    signatures_train = np.column_stack([marker_scores[name][idx[training]] for name in CAF_SIGNATURES])
    mean, sd = signatures_train.mean(0), np.maximum(signatures_train.std(0), .05)
    signature_z = (np.column_stack([marker_scores[n][idx] for n in CAF_SIGNATURES])-mean)/sd
    # Equal contribution from each patient within a state when naming programs.
    cluster_scores = np.zeros((count,len(CAF_SIGNATURES)))
    annotations = []
    for j in range(count):
        patient_values = [signature_z[training][(chosen['labels']==j)&(group==p)].mean(0)
            for p in np.unique(group) if ((chosen['labels']==j)&(group==p)).sum() >= 5]
        cluster_scores[j] = np.mean(patient_values,axis=0)
        annotations.append(dict(cluster=j, name='state_'+str(j), training_cells=chosen['sizes'][j],
            training_donors_with_5_cells=chosen['donors_with_5_cells'][j],
            descriptive_signature_z=dict(zip(CAF_SIGNATURES,cluster_scores[j].tolist()))))
    latent = memberships @ cluster_scores
    lm, ls = latent[training].mean(0), np.maximum(latent[training].std(0), .05)
    latent = (latent-lm)/ls
    ecm_cluster = int(cluster_scores[:,0].argmax())
    ecm_enriched = bool(cluster_scores[ecm_cluster,0] >= .25)
    stable = []
    for rep in range(10):
        unique = np.unique(group)
        boot = rng.choice(unique,len(unique),replace=True)
        selected = np.concatenate([np.flatnonzero(group==p) for p in boot])
        # Repeated donor draws are separate bootstrap copies, balanced by donor.
        boot_groups = np.concatenate([np.repeat(str(j),int((group==p).sum())) for j,p in enumerate(boot)])
        new_centers,_ = cluster(x[selected], boot_groups, kappa,count,seed+fold*1000+rep,starts=1)
        stable.append(float(adjusted_rand_score(chosen['labels'][sample],dist(x[sample],new_centers,kappa).argmin(1))))
    frame = pd.DataFrame(dict(global_idx=idx, barcode=atlas['barcodes'][idx], patient=groups,
        role=np.where(training,'train','test'), fold=fold, seed=seed, local_cluster=labels,
        max_soft_membership=memberships.max(1),
        ECM_enriched_state=(labels==ecm_cluster) if ecm_enriched else np.zeros(len(idx),bool),
        author_minor_for_evaluation_only=atlas['celltype_minor'][idx]))
    joint_cutoff=np.quantile(latent[training,:2],2/3,axis=0)
    frame['ECM_inflammatory_joint_high']=(latent[:,0]>=joint_cutoff[0])&(latent[:,1]>=joint_cutoff[1])
    for j,name in enumerate(CAF_SIGNATURES):
        frame['latent_'+name]=latent[:,j]
        frame['expression_'+name]=signature_z[:,j]
    for column in ('nCount_RNA','nFeature_RNA','percent.mito'):
        frame[column]=meta.iloc[idx][column].to_numpy()
    frame.to_csv(target/'caf_states.csv',index=False)
    np.savez_compressed(target/'state_model.npz',centers_lorentz=centers,curvature_k=kappa,
        cluster_program_scores=cluster_scores,temperature=temperature,
        signature_mean=mean,signature_sd=sd,latent_mean=lm,latent_sd=ls)
    test=~training
    write(target/'state_status.json',dict(stage='complete',selected_states=count,
        training_silhouette=chosen['silhouette'],bootstrap_donor_ARI=stable,
        bootstrap_donor_ARI_mean=float(np.mean(stable)),ECM_enriched_state_available=ecm_enriched,
        mixed_state_definition='Both latent ECM and inflammatory scores >= training CAF upper-tertile thresholds; no forced minimum frequency',
        mixed_state_thresholds=joint_cutoff.tolist(),
        state_annotations=annotations,selection='training CAF Lorentz silhouette, support >=30 cells and >=3 donors with >=5 cells; k=2..6',
        test_state_vs_patient_AMI=float(adjusted_mutual_info_score(groups[test],labels[test])),
        test_state_vs_author_minor_AMI=float(adjusted_mutual_info_score(atlas['celltype_minor'][idx[test]],labels[test])),
        label_evaluation_only=True,markers_are_posthoc_and_not_independent_validation=True))
    edges=pd.read_csv(target/'patient_program_edges.csv')
    h=data['membership']
    rows=[]
    for p in np.unique(groups):
        select=groups==p
        for j,name in enumerate(np.load(BASE/f'state/fold_{fold:02d}/biostate.npz')['program_names'].astype(str)):
            w=h[select,j]
            active=w>0
            for sig in CAF_SIGNATURES:
                values=marker_scores[sig][idx[select]]
                rows.append(dict(patient=p,role=frame.loc[select,'role'].iloc[0],program=name,
                    signature=sig,caf_support=int(active.sum()),
                    weighted_expression=float(np.average(values,weights=w)) if w.sum()>0 else None,
                    note='Expression overlaps model input; descriptive, not independent enrichment proof'))
    pd.DataFrame(rows).to_csv(target/'edge_function_scores.csv',index=False)
    return frame.loc[test].copy()


def residual(a, design):
    return a-design@np.linalg.lstsq(design,a,rcond=None)[0]


def correlation_table(frame, seed):
    tests=[('latent_ECM','macrophage_TREM2','macrophage_cells'),
           ('latent_ECM','CD8_exhaustion_related','CD8_cells'),
           ('latent_inflammatory','CD8_cytotoxic','CD8_cells')]
    results=[]
    for j,(xname,yname,nname) in enumerate(tests):
        d=frame[(frame.caf_cells>=30)&(frame[nname]>=30)].dropna(subset=[xname,yname]).copy()
        record=dict(seed=seed,predictor=xname,outcome=yname,patients=len(d),patient_ids=d.patient.tolist(),
                    minimum_cells_each_population=30)
        dummy=pd.get_dummies(d.subtype,drop_first=True,dtype=float).to_numpy()
        nuisance=np.column_stack([np.ones(len(d)),dummy,np.log1p(d.caf_cells),np.log1p(d[nname]),d.caf_mean_log_umi])
        if len(d)-np.linalg.matrix_rank(nuisance)<8:
            record.update(status='insufficient_patient_degrees_of_freedom')
            results.append(record);continue
        def estimate(table):
            dt=pd.get_dummies(table.subtype,drop_first=True,dtype=float).to_numpy()
            design=np.column_stack([np.ones(len(table)),dt,np.log1p(table.caf_cells),np.log1p(table[nname]),table.caf_mean_log_umi])
            a=residual(rankdata(table[xname]),design);b=residual(rankdata(table[yname]),design)
            denom=np.linalg.norm(a)*np.linalg.norm(b)
            return float(a@b/denom) if denom>1e-10 else np.nan
        value=estimate(d)
        if not np.isfinite(value):
            record.update(status='degenerate_residual_variance');results.append(record);continue
        rx=residual(rankdata(d[xname]),nuisance);ry=residual(rankdata(d[yname]),nuisance)
        rng=np.random.default_rng(seed+710+j)
        greater=0
        # Restricted residual permutations within subtype: approximate conditional
        # association test, not causal inference or an exact randomized test.
        blocks=[np.flatnonzero(d.subtype.to_numpy()==s) for s in d.subtype.unique()]
        for _ in range(4999):
            perm=np.arange(len(d))
            for block in blocks: perm[block]=rng.permutation(block)
            b=residual(ry[perm],nuisance)
            perm_value=float(rx@b/max(np.linalg.norm(rx)*np.linalg.norm(b),1e-12))
            greater+=abs(perm_value)>=abs(value)
        boot=[]
        for _ in range(2000):
            sample=pd.concat([g.iloc[rng.integers(0,len(g),len(g))] for _,g in d.groupby('subtype')])
            estimate_value=estimate(sample)
            if np.isfinite(estimate_value):boot.append(estimate_value)
        record.update(status='complete',partial_rank_correlation=value,
            permutation_p=(greater+1)/5000,bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist() if boot else None,
            bootstrap_valid=len(boot),covariates=['clinical subtype','log CAF cell count','log target cell count','CAF mean log UMI'],
            uncertainty_scope='Patient bootstrap conditional on frozen models and OOF state fitting; excludes model-refit uncertainty',
            causal=False,permutation='Approximate subtype-restricted residual permutation, 4999 draws')
        results.append(record)
    valid=[r for r in results if r['status']=='complete']
    for r,q in zip(valid,bh([r['permutation_p'] for r in valid])):
        r['BH_q_three_prespecified_associations']=float(q)
    return results


def pseudobulk(oof,counts,genes,seed,gmt):
    """Paired donor-level expression follow-up, descriptive cluster selection."""
    paired=[];patient_ids=[];raw_rows=[];descriptions=[]
    for p,group in oof.groupby('patient'):
        a=group.loc[group.ECM_enriched_state,'global_idx'].to_numpy(int)
        b=group.loc[~group.ECM_enriched_state,'global_idx'].to_numpy(int)
        if min(len(a),len(b))<20:continue
        rows=[]
        for label,ids in [('ECM_enriched',a),('other_CAF',b)]:
            total=np.asarray(counts[ids].sum(0)).ravel().astype(float)
            raw_rows.append(total);descriptions.append(dict(patient=p,state=label,cells=len(ids)))
            rows.append(np.log2(1+1e6*total/max(total.sum(),1)))
        paired.append(rows[0]-rows[1]);patient_ids.append(p)
    folder=ROOT/f'summary/seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    if raw_rows:
        sp.save_npz(folder/'pseudobulk_counts.npz',sp.csr_matrix(np.asarray(raw_rows)))
        pd.DataFrame(descriptions).to_csv(folder/'pseudobulk_samples.csv',index=False)
    if len(paired)<5:
        write(folder/'pseudobulk_status.json',dict(stage='insufficient_paired_patients',patients=len(paired),minimum=5));return
    delta=np.asarray(paired);p=np.ones(len(genes));valid=np.abs(delta).sum(0)>0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p[valid]=wilcoxon(delta[:,valid],axis=0,zero_method='wilcox',method='approx').pvalue
    p=np.nan_to_num(p,nan=1.)
    table=pd.DataFrame(dict(gene=genes,mean_paired_log2CPM_difference=delta.mean(0),
        donor_direction_fraction=(delta>0).mean(0),paired_patients=len(paired),p=p,q=bh(p)))
    table.sort_values('q').to_csv(folder/'pseudobulk_gene_results.csv',index=False)
    # Uniform background: detected in >=20% CAF training cells in every fold is
    # not available here. Use expression detected in this paired CAF pseudobulk
    # and explicitly mark this downstream analysis as exploratory.
    background=set(np.char.upper(genes[np.asarray(raw_rows).sum(0)>0]).tolist())
    hit=set(table.loc[(table.q<.05)&(table.mean_paired_log2CPM_difference>.5),'gene'].str.upper())&background
    records=[]
    for line in gmt.read_text().splitlines():
        fields=line.split('\t');members={g.split(',')[0].upper() for g in fields[2:]}&background
        overlap=hit&members
        records.append(dict(program=fields[0],background_genes=len(background),selected_genes=len(hit),
            program_genes=len(members),overlap=len(overlap),overlap_genes=';'.join(sorted(overlap)),
            p=float(hypergeom.sf(len(overlap)-1,len(background),len(members),len(hit)))))
    enrichment=pd.DataFrame(records);enrichment['q']=bh(enrichment.p)
    enrichment.sort_values('q').to_csv(folder/'hallmark_overrepresentation.csv',index=False)
    write(folder/'pseudobulk_status.json',dict(stage='complete',paired_patients=patient_ids,
        expression_unit='Patient-state summed counts; paired two-sided Wilcoxon on log2(1+CPM)',
        multiple_testing='BH across all measured genes; separate BH across 50 Hallmark programs',
        interpretation='Exploratory post-selection expression characterization; inputs and Hallmark overlap model training, not independent discovery validation'))


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    note('input_audit')
    with np.load(BASE/'data/gse176078_brca.npz') as d:atlas={k:d[k].copy() for k in d.files}
    raw=MEDIUM.parent/'data/biomed/Wu_etal_2021_BRCA_scRNASeq'
    meta=pd.read_csv(raw/'metadata.csv',low_memory=False)
    assert np.array_equal(meta.iloc[:,0].to_numpy(str),atlas['barcodes'])
    genes=np.load(BASE/'data/gse176078_brca_genes.npy').astype(str)
    counts=sp.load_npz(BASE/'data/gse176078_brca_counts.npz').tocsr()
    assert counts.shape==(len(meta),len(genes))
    expression=counts.multiply((1e4/np.maximum(np.asarray(counts.sum(1)).ravel(),1))[:,None]).tocsr()
    expression.data=np.log1p(expression.data)
    score,coverage=signatures(expression,genes)
    caf=atlas['class_names'][atlas['labels']]=='CAFs'
    plan=dict(dataset='GSE176078',all_cells=len(meta),CAF_cells=int(caf.sum()),patients=len(np.unique(atlas['patient_ids'])),
        version='Existing full-cohort state_hyper_cl: nonlinear Hallmark state hypergraph + cross-training-patient state alignment CL',
        manuscript_relation='Biological adaptation of CLhypformer; not identical to manuscript asymmetric topology-view objective',
        existing_full_model_only=True,comparative_model_training=False,folds=list(range(1,6)),seeds=list(SEEDS),
        independent_unit='patient, not cell, fold, or model seed',signatures=SIGNATURES,signature_coverage=coverage,
        latent_states='Lorentz geodesic assignment and normalized extrinsic Lorentz centers, training patients only',
        fine_labels='Author broad CAF annotations select analysis population; minor annotations used only after state fit for descriptive evaluation',
        missing_data='No imputation of patients with <30 CAF or <30 relevant immune cells; list exclusions for each association',
        primary_association='OOF latent ECM program score in CAF vs macrophage TREM2 expression',
        secondary_associations=['OOF latent ECM vs CD8 exhaustion-related expression','OOF latent inflammatory vs CD8 cytotoxic expression'],
        selection='No test association or test minor labels choose clusters or parameters; k=2..6 support and silhouette use training CAF only',
        no_forced_outcome=True,validation='External cohort or spatial confirmation is a later conditional stage; neither downloaded nor claimed validated here',
        limitations=['Existing embeddings used all expression genes. Marker validation and enrichment are descriptive, not held-out-gene evidence.',
            'Patient-program edges are within patients; repeated function across patients is tested, not direct cross-patient incidence.',
            'Per-fold coordinates and cluster IDs are never concatenated as one aligned coordinate system.',
            'State-program scores are calibrated on training patients using predeclared markers; their biology is annotation-guided.',
            'Associations are exploratory, observational and conditional on a fixed fitted pipeline; no causal or functional proof.',
            'No new-subtype, differentiation direction, treatment response or survival claims from these data.'],
        source_sha256=sha(__file__),source_metadata_sha256=sha(raw/'metadata.csv'),source_counts_sha256=sha(BASE/'data/gse176078_brca_counts.npz'),
        full_model_plan_sha256=sha(BASE/'runs/plan.json'),references=[
            'https://www.nature.com/articles/s41588-021-00911-1',
            'https://www.nature.com/articles/s41467-024-47068-z'])
    write(ROOT/'plan.json',plan)
    shutil.copy2(__file__,ROOT/Path(__file__).name)
    outcomes=[]
    for seed in SEEDS:
        frames=[]
        for fold in range(1,6):
            frame=analyze_case(fold,seed,atlas,meta,score)
            if frame is not None:frames.append(frame)
        if len(frames)!=5:
            write(ROOT/f'summary/seed_{seed}/status.json',dict(stage='incomplete_reproducible_partition',successful_folds=len(frames),
                reason='Do not pool a selected subset of successful folds'));continue
        oof=pd.concat(frames,ignore_index=True).sort_values('global_idx')
        if not np.array_equal(oof.global_idx.to_numpy(),np.flatnonzero(caf)):
            raise ValueError('OOF CAF coverage must be exactly once, with no duplicates')
        folder=ROOT/f'summary/seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
        oof.to_csv(folder/'caf_out_of_fold.csv',index=False)
        patients=[]
        for patient,table in oof.groupby('patient'):
            selected=atlas['patient_ids']==patient
            minor=atlas['celltype_minor']
            mac=np.flatnonzero(selected&(minor=='Macrophage'))
            cd8=np.flatnonzero(selected&(minor=='T cells CD8+'))
            subtype=np.unique(atlas['subtypes'][selected])
            if len(subtype)!=1:raise ValueError('Ambiguous patient clinical subtype')
            record=dict(patient=patient,subtype=subtype[0],caf_cells=len(table),macrophage_cells=len(mac),CD8_cells=len(cd8),
                caf_mean_log_umi=float(np.log1p(table.nCount_RNA).mean()),
                caf_mean_mito=float(table['percent.mito'].mean()),
                ECM_enriched_state_fraction=float(table.ECM_enriched_state.mean()),
                ECM_inflammatory_joint_high_fraction=float(table.ECM_inflammatory_joint_high.mean()))
            for name in CAF_SIGNATURES:record['latent_'+name]=float(table['latent_'+name].mean())
            for name in ('macrophage_TREM2','macrophage_lipid','CD8_cytotoxic','CD8_exhaustion_related'):
                ids=mac if name.startswith('macrophage') else cd8
                record[name]=float(score[name][ids].mean()) if len(ids) else np.nan
            patients.append(record)
        patient_frame=pd.DataFrame(patients)
        patient_frame.to_csv(folder/'patient_summary.csv',index=False)
        note('patient_associations',seed=seed,patients=len(patient_frame))
        associations=correlation_table(patient_frame,seed)
        write(folder/'associations.json',associations)
        outcomes.extend(associations)
        pseudobulk(oof,counts,genes,seed,BASE/'state/MSigDB_Hallmark_2020.gmt')
        # All OOF patients appear once. Compare biological profiles, never raw
        # coordinates from independently fitted fold representations.
        functional=pd.concat([pd.read_csv(ROOT/f'cases/fold_{f:02d}/seed_{seed}/edge_function_scores.csv') for f in range(1,6)],ignore_index=True)
        functional=functional[(functional.role=='test')&(functional.caf_support>=10)]
        recurring=functional.groupby(['program','signature']).agg(
            patients=('patient','nunique'),mean_expression=('weighted_expression','mean'),
            median_expression=('weighted_expression','median'),between_patient_sd=('weighted_expression','std')).reset_index()
        recurring['recurrent_support_at_least_3_patients']=recurring.patients>=3
        recurring.to_csv(folder/'recurrent_program_functions.csv',index=False)
    write(ROOT/'summary/association_seed_sensitivity.json',dict(results=outcomes,
        independent_patients=26,seeds_are_sensitivity_not_independent_patients=True))
    (ROOT/'README.md').write_text('''# CAF biological case study: exploratory stage\n\nUses the existing complete biological model only; no model ablation or new classifier fitting.\n\nRead plan.json before interpreting results. Inspect all case state_status.json files for failures, bootstrap stability and patient confounding. State cluster IDs and coordinates are fold-specific. Only train-annotated functional scores are combined across held-out folds.\n\nMain outputs: cases/*/*/caf_states.csv, patient_program_edges.csv, edge_function_scores.csv, caf_representations.npz; summary/seed_*/patient_summary.csv, associations.json, pseudobulk_gene_results.csv, hallmark_overrepresentation.csv.\n\nNo final manuscript figures are generated in this initial stage. Spatial and independent-cohort validation remain pending; existing all-gene embeddings do not establish held-out-marker independence.\n\nAn absent association or unstable state partition is retained and is not repaired by outcome-based tuning.\n''',encoding='utf-8')
    note('complete_exploratory_stage',associations_computed=sum(r['status']=='complete' for r in outcomes),
         next_stage='Review stability and biological candidates before independent-cohort or spatial validation; not launched')


def self_test():
    rng=np.random.default_rng(17);kappa=3.
    spatial=np.r_[rng.normal(-1,.04,(70,3)),rng.normal(1,.04,(70,3))]
    points=np.column_stack([np.sqrt(kappa+(spatial**2).sum(1)),spatial])
    groups=np.tile(np.repeat(['a','b','c','d','e','f','g'],10),2)
    centers,labels=cluster(points,groups,kappa,2,17)
    assert adjusted_rand_score(np.repeat([0,1],70),labels)>.99
    assert np.allclose(-centers[:,0]**2+(centers[:,1:]**2).sum(1),-kappa)
    d=dist(points[:10],points[:10],kappa)
    assert np.allclose(d,d.T) and np.isfinite(d).all()
    assert np.allclose(bh([.01,.04,.03]),[.03,.04,.04])
    print(json.dumps(dict(synthetic_Lorentz_state_recovery=True,curvature=-1/kappa,BH_check=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        try:run()
        except Exception as error:
            note('failed',error=repr(error));raise
