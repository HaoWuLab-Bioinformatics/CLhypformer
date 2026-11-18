"""Train-patient program discovery followed by gated, frozen cross-patient retrieval.

Repeated NMF is deliberately not described as the official cNMF workflow.
Validation drives all screening. No test/external evaluation before the screen.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, hypergeom
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from gse_caf_analysis_20261017 import BASE, sha
from gse_caf_pca_compare_20261018 import ROOT as PCA_ROOT, normalize
from gse_tumor_state_runtime_20261022 import source_data, source_export, SEEDS

MEDIUM = Path(__file__).resolve().parent
ROOT = BASE.parent/'gse176078_patient_programs_20261022'
RANDOM = 20261022
CONFIG = dict(reference_fold=1, random_seed=RANDOM, seeds=list(SEEDS), candidates_K=[3,4,5], repetitions=20,
              minimum_discovery_cells=200, variable_A_genes=2000, B_gene_fraction=.2,
              maximum_iterations=300, tolerance=.001, minimum_program_patients=3,
              minimum_validation_cells=30, minimum_B_genes=10,
              validation_positive_patients=3, retrieval_k=[10,30], matched_dimension=30,
              primary='patient-mean QC-adjusted B-score retrieval Spearman; matched30, k10',
              validation_only_screen=True, classifier_retraining=False,
              interpretation='Expression consistency, not unseen-gene prediction or causal mechanism')


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def event(stage, **kw):
    obj = {'time':datetime.now(timezone.utc).isoformat(), 'stage':stage, **kw}
    write(ROOT/'status.json', obj)
    print(json.dumps(obj), flush=True)


def stable_hash(text):
    return int.from_bytes(hashlib.sha256(str(text).encode()).digest()[:8], 'big')


def rho(x, y):
    if len(x) < 10 or np.std(x) < 1e-7 or np.std(y) < 1e-7:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else None


def residual(x, qc):
    q = (qc-qc.mean(0))/np.maximum(qc.std(0), 1e-6)
    design = np.column_stack([np.ones(len(q)), q])
    return x-design@np.linalg.lstsq(design, x, rcond=None)[0]


def transform_usage(x, dictionary):
    if dictionary.shape[0] == 0:
        return np.empty((len(x), 0), np.float32)
    x = np.asarray(x, np.float32)
    h = np.asarray(dictionary, np.float32)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always', ConvergenceWarning)
        w, _, iterations = non_negative_factorization(x, W=np.maximum(x@h.T, 1e-6), H=h.copy(),
            n_components=len(h), init='custom', update_H=False, solver='cd',
            max_iter=500, tol=1e-4, random_state=RANDOM)
    if not np.isfinite(w).all():
        raise FloatingPointError('Nonfinite fixed-dictionary usage')
    return w.astype(np.float32)


def consensus(hs):
    """Medoid initialization, Hungarian component alignment, average unit loadings."""
    unit = hs/np.maximum(np.linalg.norm(hs, axis=2, keepdims=True), 1e-10)
    scores = np.zeros((len(hs), len(hs)))
    for a in range(len(hs)):
        for b in range(a):
            sim = unit[a]@unit[b].T
            i, j = linear_sum_assignment(-sim)
            scores[a,b] = scores[b,a] = sim[i,j].mean()
    medoid = int(scores.sum(1).argmax())
    aligned, similarity = [], []
    for h in unit:
        sim = unit[medoid]@h.T
        i, j = linear_sum_assignment(-sim)
        order = j[np.argsort(i)]
        aligned.append(h[order]); similarity.append(sim[np.arange(len(h)),order])
    average = np.mean(aligned, axis=0)
    average /= np.maximum(np.linalg.norm(average, axis=1, keepdims=True), 1e-10)
    return average.astype(np.float32), np.mean(similarity, axis=0)


def prepare():
    event('prepare_train_only_gene_split')
    data = source_data()
    with np.load(source_export(SEEDS[0])) as z:
        idx, role = z['idx'].copy(), z['role'].copy()
    genes = np.load(BASE/'data/gse176078_brca_genes.npy').astype(str)
    counts = sp.load_npz(BASE/'data/gse176078_brca_counts.npz').tocsr()[idx]
    expression = normalize(counts)
    del counts
    patients = data['patient_ids'][idx].astype(str)
    train = role == 0
    mean, second, prevalence = np.zeros(len(genes)), np.zeros(len(genes)), np.zeros(len(genes))
    for p in np.unique(patients[train]):
        z = expression[train & (patients == p)]
        mean += np.asarray(z.mean(0)).ravel()
        second += np.asarray(z.power(2).mean(0)).ravel()
        prevalence += np.asarray((z > 0).mean(0)).ravel() >= .01
    n = len(np.unique(patients[train])); mean /= n; second /= n
    sd = np.sqrt(np.maximum(second-mean**2, 0))
    allowed = np.array([not g.upper().startswith(('MT-','RPL','RPS','HBA','HBB')) for g in genes])
    allowed &= (prevalence >= 3) & (sd >= .05)
    is_b = np.array([stable_hash('panelB:'+g)%5 == 0 for g in genes])
    a = np.flatnonzero(allowed & ~is_b)
    a = a[np.argsort(sd[a]**2)[::-1]][:CONFIG['variable_A_genes']]
    b = np.flatnonzero(allowed & is_b)
    if len(a) < 500 or len(b) < 100:
        raise RuntimeError('Insufficient eligible A/B genes')
    meta = pd.read_csv(BASE.parent.parent/'data/biomed/Wu_etal_2021_BRCA_scRNASeq/metadata.csv', index_col=0)
    meta = meta.loc[data['barcodes'][idx]]
    qc = np.column_stack([np.log1p(meta.nCount_RNA.to_numpy()), meta['percent.mito'].to_numpy()])
    x = np.asarray(expression[:,a].toarray(), np.float32)/np.maximum(sd[a], .1).astype(np.float32)
    np.savez_compressed(ROOT/'gene_preprocessing.npz', idx=idx, role=role, A=a, B=b,
                        training_mean=mean, training_sd=sd)
    write(ROOT/'gene_split.json', dict(A=genes[a].tolist(), B=genes[b].tolist(),
          split='SHA256(panelB:gene) modulo 5; preselection independent of validation/test',
          limitation='B excluded from new NMF fitting; may be present in frozen upstream model inputs'))
    return data, idx, role, patients, genes, expression, x, a, b, mean, sd, qc


def discover(x, patients, training):
    components, records = [], []
    for patient in np.unique(patients[training]):
        mask = training & (patients == patient)
        if mask.sum() < CONFIG['minimum_discovery_cells']:
            records.append(dict(patient=patient, cells=int(mask.sum()), eligible=False,
                                reason='Below 200 discovery cells; retained for transfer'))
            continue
        local = x[mask]
        candidates = []
        for k in CONFIG['candidates_K']:
            path = ROOT/f'nmf/{patient}/K{k}.npz'
            if not path.exists():
                event('patient_NMF', patient=patient, cells=len(local), K=k, repetitions=CONFIG['repetitions'])
                hs, errors, iterations = [], [], []
                for run in range(CONFIG['repetitions']):
                    model = NMF(n_components=k, init='random', random_state=(RANDOM+stable_hash(patient)%100000+k*100+run),
                                max_iter=CONFIG['maximum_iterations'], tol=CONFIG['tolerance'], solver='cd')
                    with warnings.catch_warnings(record=True):
                        warnings.simplefilter('always', ConvergenceWarning)
                        model.fit_transform(local)
                    hs.append(model.components_); errors.append(model.reconstruction_err_); iterations.append(model.n_iter_)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, H=np.stack(hs), errors=errors, iterations=iterations)
            with np.load(path) as saved:
                h, stability = consensus(saved['H'])
                error = float(np.mean(saved['errors']))
                fraction_at_limit = float(np.mean(saved['iterations'] >= CONFIG['maximum_iterations']))
            # Null calibrated on training gene permutations only, not held-out scores.
            rng = np.random.default_rng(RANDOM+stable_hash(patient)%100000+k)
            null = np.array([h[t%k]@h[(t+1)%k, rng.permutation(h.shape[1])] for t in range(500)])
            threshold = max(.8, float(np.quantile(null, .99)))
            robust = stability >= threshold
            candidates.append(dict(K=k, h=h, stability=stability, robust=robust,
                stability_threshold=threshold, error=error, fraction_at_limit=fraction_at_limit))
        eligible = [c for c in candidates if c['robust'].sum() >= 2 and c['fraction_at_limit'] <= .5]
        if eligible:
            best_error = min(c['error'] for c in eligible)
            chosen = min([c for c in eligible if c['error'] <= best_error*1.05], key=lambda c:c['K'])
            for j in np.flatnonzero(chosen['robust']):
                vector = chosen['h'][j]
                usage = transform_usage(local, vector[None,:])[:,0]
                if np.std(usage) <= .05:
                    continue
                components.append(vector)
                records.append(dict(patient=patient, cells=int(mask.sum()), eligible=True,
                    K=chosen['K'], component=int(j), stability=float(chosen['stability'][j]),
                    stability_threshold=chosen['stability_threshold'], component_id=len(components)-1,
                    fraction_initializations_at_limit=chosen['fraction_at_limit']))
        else:
            records.append(dict(patient=patient, cells=int(mask.sum()), eligible=False,
                reason='No K with >=2 stable components and <=50% iterations capped'))
        write(ROOT/'nmf'/patient/'selection.json', dict(candidates=[{k:(v.tolist() if isinstance(v,np.ndarray) else v)
            for k,v in c.items() if k != 'h'} for c in candidates],
            selected_K=chosen['K'] if eligible else None,
            selection='Training reconstruction within 5% best among stable candidates; then smallest K'))
    pd.DataFrame(records).to_csv(ROOT/'patient_program_audit.csv', index=False)
    if not components:
        return np.empty((0,x.shape[1]), np.float32), []
    h = np.stack(components)
    membership = np.zeros_like(h, dtype=bool)
    for j, v in enumerate(h):
        order = np.argsort(v)[::-1][:50]
        membership[j,order[v[order] > 0]] = True
    distances = squareform(pdist(membership, metric='jaccard'))
    rng = np.random.default_rng(RANDOM)
    null = []
    for _ in range(2000):
        aa, bb = rng.choice(h.shape[1],50,False), rng.choice(h.shape[1],50,False)
        intersect = len(np.intersect1d(aa,bb)); null.append(intersect/(100-intersect))
    cutoff = max(.1, float(np.quantile(null,.99)))
    labels = fcluster(linkage(squareform(distances, checks=False), method='complete'), 1-cutoff, criterion='distance') if len(h)>1 else np.array([1])
    component_records = [r for r in records if r['eligible']]
    programs, info = [], []
    for label in np.unique(labels):
        ids = np.flatnonzero(labels==label)
        # One component per patient, highest stability; patient equal weighting.
        chosen = {}
        for i in ids:
            rec = component_records[i]
            if rec['patient'] not in chosen or rec['stability'] > component_records[chosen[rec['patient']]]['stability']:
                chosen[rec['patient']] = int(i)
        if len(chosen) < CONFIG['minimum_program_patients']:
            continue
        vector = h[list(chosen.values())].mean(0)
        vector /= np.maximum(np.linalg.norm(vector), 1e-10)
        programs.append(vector)
        info.append(dict(program=len(programs)-1, patients=sorted(chosen), n_patients=len(chosen),
                         component_ids=list(chosen.values()), top50_Jaccard_threshold=cutoff))
    write(ROOT/'program_discovery_audit.json', dict(patient_components=len(h), groups=len(np.unique(labels)),
          shared_programs=len(programs), null_Jaccard_99=float(np.quantile(null,.99)), cutoff=cutoff,
          programs=info, matching='Complete-linkage top50 Jaccard; >=3 patients, equal patient averaging'))
    return np.stack(programs) if programs else np.empty((0,x.shape[1]), np.float32), info


def select_B(expression, b, mean, sd, usage, patients, train, qc):
    """Annotate A-derived programs using B associations in TRAIN patients only."""
    y = expression[:,b]
    effects, supports = [], []
    for patient in np.unique(patients[train]):
        mask = train & (patients == patient)
        if mask.sum() < 30:
            continue
        # Dense blocks here are restricted to one training patient, not the atlas.
        genes = np.asarray(y[mask].toarray(), np.float64)
        genes = residual(genes, qc[mask])
        u = residual(np.log1p(usage[mask]), qc[mask])
        denom = np.maximum(np.linalg.norm(genes,axis=0)[:,None]*np.linalg.norm(u,axis=0)[None,:],1e-10)
        cor = (genes.T@u)/denom
        effects.append(cor); supports.append(np.std(u,axis=0) > .02)
    effects = np.stack(effects); supports = np.stack(supports)
    panels = []
    for j in range(usage.shape[1]):
        valid = supports[:,j]
        if valid.sum()<3:
            panels.append([]); continue
        associations = effects[valid,:,j]
        effect = associations.mean(0)
        keep = (effect > .1) & ((associations>0).mean(0)>=.6)
        ids = np.flatnonzero(keep)
        ids = ids[np.argsort(effect[ids])[::-1]][:20]
        panels.append(b[ids].tolist())
    scores = np.zeros((expression.shape[0], usage.shape[1]), np.float32)
    for j, panel in enumerate(panels):
        if panel:
            values = np.asarray(expression[:,panel].toarray())
            scores[:,j] = ((values-mean[panel])/np.maximum(sd[panel],.1)).mean(1)
    return panels, scores


def annotate_programs(dictionary, genes, a):
    """Training-A-only functional annotation; never a program selection criterion."""
    gene_sets = {}
    for line in (BASE/'state/MSigDB_Hallmark_2020.gmt').read_text().splitlines():
        fields = line.split('\t')
        gene_sets[fields[0]] = {g.split(',')[0].upper() for g in fields[2:]}
    universe = set(genes[a].tolist()); rows=[]
    for j, vector in enumerate(dictionary):
        top = set(genes[a[np.argsort(vector)[::-1][:50]]].tolist())
        for name, members in gene_sets.items():
            present = members & universe
            if len(present)<3:
                continue
            overlap = top & present
            rows.append(dict(program=j, Hallmark=name, overlap=len(overlap),
                eligible_set_genes=len(present), eligible_A_genes=len(universe),
                p=float(hypergeom.sf(len(overlap)-1,len(universe),len(present),len(top))),
                overlapping_genes=';'.join(sorted(overlap))))
    if rows:
        frame=pd.DataFrame(rows)
        order=np.argsort(frame.p.to_numpy())
        p=frame.p.to_numpy()[order]
        adj=np.minimum.accumulate((p*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
        q=np.zeros(len(p));q[order]=np.minimum(adj,1)
        frame['BH_q']=q
        frame.to_csv(ROOT/'training_program_Hallmark_annotation.csv',index=False)


def validate_programs(usage, scores, panels, patients, role, qc):
    rows, gates = [], []
    for j in range(usage.shape[1]):
        for patient in np.unique(patients[role==1]):
            mask = (role==1) & (patients==patient)
            if mask.sum()<CONFIG['minimum_validation_cells'] or len(panels[j])<CONFIG['minimum_B_genes']:
                continue
            raw = rho(np.log1p(usage[mask,j]), scores[mask,j])
            adjusted = rho(residual(np.log1p(usage[mask,j]),qc[mask]),residual(scores[mask,j],qc[mask]))
            rows.append(dict(program=j, patient=patient, cells=int(mask.sum()), role='validation',
                             rho=raw, qc_rho=adjusted))
        current = [r for r in rows if r['program']==j]
        positive = [r for r in current if r['rho'] is not None and r['qc_rho'] is not None
                    and r['rho']>0 and r['qc_rho']>0]
        gates.append(dict(program=j, B_genes=len(panels[j]), validation_patients=len(current),
            positive_validation_patients=len(positive), passed=len(positive)>=CONFIG['validation_positive_patients']))
    pd.DataFrame(rows, columns=['program','patient','cells','role','rho','qc_rho']).to_csv(ROOT/'validation_program_consistency.csv',index=False)
    write(ROOT/'program_validation_gates.json', gates)
    return [g['program'] for g in gates if g['passed']]


def fit_coordinates(x, groups, dimension):
    unique, code, n = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1/n[code]; weights /= weights.sum()
    mean = np.sum(x*weights[:,None],axis=0)
    sd = np.sqrt(np.sum((x-mean)**2*weights[:,None],axis=0))
    sd = np.maximum(sd,1e-6)
    z = (x-mean)/sd
    if dimension is None:
        return dict(mean=mean, sd=sd, components=np.eye(x.shape[1]))
    values, vectors = np.linalg.eigh(z.T@(z*weights[:,None]))
    return dict(mean=mean, sd=sd, components=vectors[:,np.argsort(values)[-dimension:][::-1]])


def project(x, params):
    return ((x-params['mean'])/params['sd'])@params['components']


def align_export(path, idx, role):
    with np.load(path) as z:
        positions = {int(v):i for i,v in enumerate(z['idx'])}
        assert len(positions)==len(idx)
        order = np.array([positions[int(i)] for i in idx])
        assert np.array_equal(z['role'][order], role)
        return z['points'][order].astype(np.float64)


def retrieve(data, idx, role, patients, expression, scores, usage, selected, qc):
    event('frozen_retrieval_validation', programs=selected)
    train, valid = role==0, role==1
    groups = np.unique(patients[train])
    per_patient = min(200, min(int((train&(patients==p)).sum()) for p in groups))
    rng = np.random.default_rng(RANDOM)
    reference = np.concatenate([rng.choice(np.flatnonzero(train&(patients==p)),per_patient,replace=False) for p in groups])
    queries = np.flatnonzero(valid)
    np.savez_compressed(ROOT/'retrieval_split.npz', reference_local=reference, query_local=queries,
        reference_source_idx=idx[reference], query_source_idx=idx[queries])
    with np.load(PCA_ROOT/'pca/fold_01.npz') as z:
        pca = {k:z[k].copy() for k in z.files}
    pcapoints = (np.asarray(expression@pca['components'].T)-pca['gene_mean']@pca['components'].T)/pca['scale']
    rows = []
    for seed in SEEDS:
        source = align_export(source_export(seed),idx,role)
        light_path = ROOT/f'representations/lighthgnn/seed_{seed}.npz'
        if not light_path.exists():
            subprocess.run(['/home/user018/miniconda3/envs/lighthgnn/bin/python',
                str(MEDIUM/'gse_patient_program_export_20261022.py'),'--seed',str(seed)],
                cwd=MEDIUM,check=True)
        light = align_export(light_path,idx,role)
        for arm, points in [('full',source),('pca',pcapoints),('lighthgnn',light)]:
            for label, dimension in [('matched30',30),('native',None)]:
                params = fit_coordinates(points[train],patients[train],dimension)
                z = project(points,params)
                out = ROOT/f'retrieval/{arm}/seed_{seed}/{label}'
                out.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(out/'train_only_projection.npz',**params)
                nn = NearestNeighbors(n_neighbors=max(CONFIG['retrieval_k'])).fit(z[reference])
                distances, local = nn.kneighbors(z[queries])
                np.savez_compressed(out/'validation_neighbors.npz',query_source_idx=idx[queries],
                     neighbor_source_idx=idx[reference[local]],distances=distances)
                for k in CONFIG['retrieval_k']:
                    neighbors = reference[local[:,:k]]
                    prediction = scores[neighbors].mean(1)
                    pred_usage = np.log1p(usage[neighbors]).mean(1)
                    for patient in np.unique(patients[queries]):
                        take = patients[queries]==patient
                        if take.sum()<CONFIG['minimum_validation_cells']:
                            continue
                        q = queries[take]
                        for program in selected:
                            rows.append(dict(arm=arm,seed=seed,dimension=label,k=k,patient=patient,
                                program=program,cells=int(take.sum()),
                                B_rho=rho(prediction[take,program],scores[q,program]),
                                B_qc_rho=rho(residual(prediction[take,program],qc[q]),residual(scores[q,program],qc[q])),
                                A_usage_rho=rho(pred_usage[take,program],np.log1p(usage[q,program])),
                                mean_neighbor_patient_count=float(np.mean([len(np.unique(patients[n])) for n in neighbors[take]]))))
    # A seeded random reference is a negative control, never a competing model.
    random_neighbors = np.random.default_rng(RANDOM+1).choice(reference,size=(len(queries),10),replace=True)
    prediction = scores[random_neighbors].mean(1)
    for patient in np.unique(patients[queries]):
        take = patients[queries]==patient; q=queries[take]
        for program in selected:
            rows.append(dict(arm='random',seed=RANDOM,dimension='control',k=10,patient=patient,
                program=program,cells=len(q),B_rho=rho(prediction[take,program],scores[q,program]),
                B_qc_rho=rho(residual(prediction[take,program],qc[q]),residual(scores[q,program],qc[q])),
                A_usage_rho=None,mean_neighbor_patient_count=None))
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT/'validation_retrieval_patient_program.csv',index=False)
    primary=frame[(frame.dimension=='matched30')&(frame.k==10)]
    perseed=primary.groupby(['arm','seed','patient']).B_qc_rho.mean().reset_index()
    perseed.to_csv(ROOT/'validation_primary_per_patient_seed.csv',index=False)
    average=perseed.groupby(['patient','arm']).B_qc_rho.mean().unstack('arm')
    results=[]
    for comparator in ('pca','lighthgnn'):
        pair=average[['full',comparator]].dropna()
        delta=(pair.full-pair[comparator]).to_numpy()
        if len(delta):
            rng=np.random.default_rng(RANDOM)
            bootstrap=delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1)
            interval=np.quantile(bootstrap,[.025,.975]).tolist()
            if len(delta)<=16:
                signs=np.array(list(itertools.product([-1,1],repeat=len(delta))))
                p=float((np.abs((signs*delta).mean(1))>=abs(delta.mean())-1e-12).mean())
            else:p=None
        else:interval=None;p=None
        seed_means=[]
        for seed in SEEDS:
            s=perseed[perseed.seed==seed].pivot(index='patient',columns='arm',values='B_qc_rho')
            d=(s['full']-s[comparator]).dropna();seed_means.append(float(d.mean()) if len(d) else None)
        passed=(len(delta)>=3 and float(delta.mean())>0 and float(np.mean(delta>0))>=2/3
                and all(v is not None and v>0 for v in seed_means))
        results.append(dict(comparator=comparator,patients=len(delta),mean_delta=float(delta.mean()) if len(delta) else None,
            positive_patient_fraction=float(np.mean(delta>0)) if len(delta) else None,
            patient_bootstrap_95_interval=interval,exact_patient_signflip_p=p,
            seed_mean_deltas=seed_means,screen_passed=bool(passed)))
    ps=[r['exact_patient_signflip_p'] for r in results]
    if all(v is not None for v in ps):
        order=np.argsort(ps);adj=np.minimum.accumulate((np.array(ps)[order]*len(ps)/np.arange(1,len(ps)+1))[::-1])[::-1]
        for i,q in zip(order,np.minimum(adj,1)):results[i]['BH_q']=float(q)
    write(ROOT/'retrieval_validation_summary.json',dict(comparisons=results,
        screen_passed=all(r['screen_passed'] for r in results),
        rule='Both comparators: positive mean, >=2/3 patients positive, both model seeds positive; >=3 patients',
        uncertainty='Development screen; small patient count, bootstrap and permutation values exploratory',
        reference_patients=len(groups),reference_cells_per_patient=per_patient,
        test_metrics_accessed=False,external_metrics_accessed=False))
    return all(r['screen_passed'] for r in results)


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'complete.json').exists():
        print((ROOT/'complete.json').read_text());return
    if (ROOT/'protocol.json').exists() and json.loads((ROOT/'protocol.json').read_text())!=CONFIG:
        raise RuntimeError('Cached analysis has a different protocol; use a new run directory')
    write(ROOT/'protocol.json',CONFIG)
    data,idx,role,patients,genes,expression,x,a,b,mean,sd,qc=prepare()
    dictionary,info=discover(x,patients,role==0)
    np.savez_compressed(ROOT/'shared_program_dictionary.npz',H=dictionary,A_gene_indices=a,A_genes=genes[a])
    write(ROOT/'shared_program_genes.json',[dict(**r,top50_genes=genes[a[np.argsort(dictionary[j])[::-1][:50]]].tolist())
          for j,r in enumerate(info)])
    annotate_programs(dictionary,genes,a)
    if len(dictionary)<2:
        finish('stopped_insufficient_shared_programs',shared_programs=len(dictionary));return
    event('fixed_dictionary_usage_and_B_validation',shared_programs=len(dictionary))
    usage=transform_usage(x,dictionary)
    panels,scores=select_B(expression,b,mean,sd,usage,patients,role==0,qc)
    write(ROOT/'B_panels.json',[dict(program=j,genes=genes[p].tolist(),
        selection='Within-training-patient QC-adjusted positive associations; not used to fit/match NMF programs') for j,p in enumerate(panels)])
    np.savez_compressed(ROOT/'source_program_usage.npz',idx=idx,role=role,usage=usage,
                         soft_membership=usage/np.maximum(usage.sum(1,keepdims=True),1e-8),B_scores=scores)
    selected=validate_programs(usage,scores,panels,patients,role,qc)
    if len(selected)<2:
        finish('stopped_validation_program_gate',shared_programs=len(dictionary),validated_programs=selected);return
    passed=retrieve(data,idx,role,patients,expression,scores,usage,selected,qc)
    finish('screen_passed_confirmation_ready' if passed else 'stopped_no_consistent_retrieval_advantage',
           shared_programs=len(dictionary),validated_programs=selected,
           next_step='Lock protocol then confirm fold3/5 and external cohort' if passed else 'Retain honest development results; no parameter search')


def finish(status,**kw):
    record=dict(status=status,time=datetime.now(timezone.utc).isoformat(),**kw,
                test_evaluation=False,external_evaluation=False,spatial_mapping=False,
                classifier_retraining=False)
    write(ROOT/'complete.json',record);event('complete',**record)


def check():
    """Mechanism and leakage tests plus real-data/old-checkpoint dependency audits."""
    ROOT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(22)
    h=np.abs(rng.normal(size=(3,60))).astype(np.float32)
    w=np.abs(rng.normal(size=(100,3))).astype(np.float32)
    recovered=transform_usage(w@h,h)
    error=float(np.linalg.norm(recovered@h-w@h)/np.linalg.norm(w@h))
    assert error<.01,error
    hs=np.stack([h[[2,0,1]],h[[1,2,0]],h])
    _,stability=consensus(hs);assert np.all(stability>.999)
    groups=np.repeat(['a','b'],50);p=fit_coordinates(w,groups,2)
    changed=np.concatenate([w,rng.normal(1e6,1,size=(10,3))])
    p2=fit_coordinates(changed[:100],groups,2)
    assert all(np.array_equal(p[k],p2[k]) for k in p)
    assert rho(np.ones(20),np.arange(20)) is None
    split=np.array([stable_hash('panelB:'+str(i))%5==0 for i in range(100)])
    assert split.any() and (~split).any()
    qc=rng.normal(size=(100,2));assert np.linalg.norm(residual(qc[:,0],qc))<1e-8
    data=source_data()
    for left,right in [('train_idx','valid_idx'),('train_idx','test_idx'),('valid_idx','test_idx')]:
        assert not set(data['patient_ids'][data[left]]) & set(data['patient_ids'][data[right]])
    dependencies=[BASE/'data/gse176078_brca_counts.npz',PCA_ROOT/'pca/fold_01.npz',
        Path('/home/user018/miniconda3/envs/lighthgnn/bin/python')]
    for seed in SEEDS:
        dependencies.extend([BASE/f'runs/state_hyper_cl/fold_01/seed_{seed}/checkpoint.pt',
            BASE.parent/f'gse176078_caf_confirm_20261022/selected/lighthgnn/fold_01/seed_{seed}/selection.json'])
    for p in dependencies:
        assert p.exists(),str(p)
    write(ROOT/'startup_check.json',dict(passed=True,NNLS_reconstruction_relative_error=error,
        component_permutation_alignment=True,heldout_projection_leakage_check=True,
        patient_disjoint_roles=True,constant_target_not_scored=True,QC_residualization=True,
        dependencies=[str(p) for p in dependencies],protocol=CONFIG))
    print(json.dumps(dict(preflight_passed=True,NNLS_reconstruction_relative_error=error)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    with threadpool_limits(limits=4):
        if parser.parse_args().check:check()
        else:
            try:
                run()
            except Exception as exc:
                event('failed',error_type=type(exc).__name__,message=str(exc))
                raise
