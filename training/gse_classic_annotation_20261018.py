"""Donor-held-out nine-class annotation baselines, official method packages."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
import argparse,json,pickle,warnings
from pathlib import Path
from datetime import datetime,timezone
import importlib.metadata
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,confusion_matrix
from hashlib import sha256
from scipy.sparse.linalg import LinearOperator, svds

BASE = Path(os.environ['CLHYP_ATLAS_ROOT']) / 'bio_runs/gse176078_full_20261017'
EXPRESSION = Path(os.environ.get('CLHYP_EXPRESSION_DIR', str(BASE / 'data')))
ROOT = Path(os.environ['CLHYP_CLASSICAL_OUTPUT'])
SEEDS = (168023000, 168023001)
FOLDS = range(1, 6)
SMOKE = False
CS = (.01, .1, 1., 10., 100.)

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def normalize(counts):
    x = counts.multiply((1e4 / np.maximum(np.asarray(counts.sum(1)).ravel(), 1))[:, None]).tocsr()
    x.data = np.log1p(x.data)
    return x

def fit_projection(x, train, fold):
    tx = x[train].astype(np.float64)
    mean = np.asarray(tx.mean(0)).ravel()
    n, g = tx.shape
    op = LinearOperator((n, g), dtype=np.float64,
        matvec=lambda v: tx @ v - np.full(n, mean @ v),
        rmatvec=lambda v: tx.T @ v - mean*v.sum(),
        matmat=lambda v: tx @ v - np.ones((n, 1)) @ (mean @ v)[None, :],
        rmatmat=lambda v: tx.T @ v - mean[:, None]*v.sum(0)[None, :])
    _, values, vectors = svds(op, k=50, which='LM', tol=1e-6, random_state=168023018+fold)
    weights = vectors[np.argsort(values)[::-1]]
    offset = mean @ weights.T
    scale = np.maximum((np.asarray(tx @ weights.T) - offset).std(0), 1e-6)
    return (np.asarray(x @ weights.T) - offset) / scale

def singler_numpy_compatibility():
    if 'bool' not in np.__dict__:
        np.bool = np.bool_
    if int(np.__version__.split('.')[0]) >= 2:
        return
    import importlib.util

    spec = importlib.util.find_spec('scranpy')
    if spec is None or any('copy=None' in source.read_text(encoding='utf-8')
                           for source in Path(spec.origin).parent.glob('*.py')):
        raise RuntimeError('Use the historical SingleR environment or an environment with NumPy 2.')

def fold_indices(fold):
    with np.load(BASE / f'data/fold_{fold:02d}/biomed/gse176078_brca.npz') as arrays:
        ids = {s: arrays[s + '_idx'] for s in ('train', 'valid', 'test')}
        labels = arrays['labels']
        names, groups, barcodes = (arrays[k].astype(str) for k in ('class_names', 'patient_ids', 'barcodes'))
    if SMOKE:
        ids = {s: np.concatenate([idx[labels[idx] == k][:24] for k in range(len(names))])
               for s, idx in ids.items()}
    return ids, labels, names, groups, barcodes

def note(stage,**values):
    value=dict(stage=stage,time=datetime.now(timezone.utc).isoformat(),**values)
    write(ROOT/'status.json',value);print(json.dumps(value),flush=True)

def metrics(y,pred,groups,names):
    rows=[]
    for donor in np.unique(groups):
        mask=groups==donor;true=y[mask];p=pred[mask]
        # Match prior full-cohort report: macro over true classes in each donor.
        observed=np.unique(true)
        precision,recall,f1,support=precision_recall_fscore_support(true,p,labels=np.arange(len(names)),zero_division=0)
        rows.append(dict(patient=str(donor),cells=int(mask.sum()),accuracy=float(accuracy_score(true,p)),
            macro_f1_observed=float(f1_score(true,p,labels=observed,average='macro',zero_division=0)),
            macro_f1_fixed9=float(f1.mean()),class_metrics={str(name):dict(precision=float(a),recall=float(b),f1=float(c),support=int(d)) for name,a,b,c,d in zip(names,precision,recall,f1,support)}))
    return rows

def save_case(method,fold,seed,split,ids,truth,pred,groups,barcodes,names,scores=None,score_names=None):
    target=ROOT/f'{method}/fold_{fold:02d}/seed_{seed}';target.mkdir(parents=True,exist_ok=True)
    if len(pred)!=len(ids):raise ValueError('Prediction length mismatch')
    np.savez_compressed(target/f'{split}_predictions.npz',global_idx=ids,barcodes=barcodes[ids],true=truth,pred=pred,
        patient_ids=groups[ids],scores=np.empty((len(ids),0)) if scores is None else scores,
        score_names=np.asarray([] if score_names is None else score_names))
    records=metrics(truth,pred,groups[ids],names)
    write(target/f'{split}_metrics.json',dict(macro_f1=float(f1_score(truth,pred,labels=np.arange(len(names)),average='macro',zero_division=0)),
        accuracy=float(accuracy_score(truth,pred)),patient_macro_f1=float(np.mean([v['macro_f1_observed'] for v in records])),
        patients=records,confusion=confusion_matrix(truth,pred,labels=np.arange(len(names))).tolist()))
    return records

def run_pca():
    # Standard PCA parameters from the fixed prior comparison; no refit on queries.
    data=normalize(sp.load_npz(EXPRESSION/'gse176078_brca_counts.npz').tocsr())
    for fold in FOLDS:
        ids, y, names, groups, barcodes = fold_indices(fold)
        x=fit_projection(data,ids['train'],fold);scaler=StandardScaler().fit(x[ids['train']]);x=scaler.transform(x)
        for seed in SEEDS:
            case=ROOT/f'pca_lr/fold_{fold:02d}/seed_{seed}';case.mkdir(parents=True,exist_ok=True)
            if (case/'complete.json').exists():continue
            note('pca_logistic_training',fold=fold,seed=seed);candidates=[];models={}
            for c in CS:
                model=LogisticRegression(C=c,solver='lbfgs',max_iter=3000,tol=1e-5,random_state=seed)
                model.fit(x[ids['train']],y[ids['train']])
                if model.n_iter_.max()>=3000:raise RuntimeError('PCA LR did not converge')
                pred=model.predict(x[ids['valid']]);score=np.mean([v['macro_f1_observed'] for v in metrics(y[ids['valid']],pred,groups[ids['valid']],names)])
                candidates.append(dict(C=c,validation_patient_macro_f1=float(score)));models[c]=model
            chosen=max(candidates,key=lambda v:(v['validation_patient_macro_f1'],-v['C']))['C'];model=models[chosen]
            write(case/'selection.json',dict(chosen_C=chosen,candidates=candidates))
            with (case/'model.pkl').open('wb') as stream:pickle.dump(dict(classifier=model,scaler=scaler),stream)
            for split in ('valid','test'):
                p=model.predict_proba(x[ids[split]]);save_case('pca_lr',fold,seed,split,ids[split],y[ids[split]],model.classes_[p.argmax(1)],groups,barcodes,names,p,names[model.classes_])
            write(case/'complete.json',dict(complete=True,seed=seed))

def run_celltypist():
    import celltypist,anndata
    # Library normalization computed across all measured genes for every cell.
    x=normalize(sp.load_npz(EXPRESSION/'gse176078_brca_counts.npz').tocsr())
    genes=np.load(EXPRESSION/'gse176078_brca_genes.npy').astype(str)
    for fold in FOLDS:
        ids, y, names, groups, barcodes = fold_indices(fold)
        for seed in SEEDS:
            case=ROOT/f'celltypist/fold_{fold:02d}/seed_{seed}';case.mkdir(parents=True,exist_ok=True)
            if (case/'complete.json').exists():continue
            note('celltypist_training',fold=fold,seed=seed)
            # Official deterministic L2 log-regression with training-only supervised
            # feature selection; no pretrained immune reference, no query voting.
            adata=anndata.AnnData(x[ids['train']].copy());adata.var_names=genes;adata.obs['label']=names[y[ids['train']]]
            np.random.seed(seed)
            model=celltypist.train(adata,labels='label',n_jobs=4,max_iter=1000,feature_selection=True,top_genes=300,
                use_SGD=False,with_mean=True,C=1.,random_state=seed)
            model.write(str(case/'model.pkl'))
            write(case/'selection.json',dict(parameters='Official train feature_selection=True, top_genes=300, C=1 fixed, max_iter=1000, no majority voting',test_tuning=False))
            for split in ('valid','test'):
                scores=[];predictions=[]
                for start in range(0,len(ids[split]),2000):
                    selected=ids[split][start:start+2000];query=anndata.AnnData(x[selected].copy());query.var_names=genes
                    result=celltypist.annotate(query,model=model,majority_voting=False)
                    labels=result.predicted_labels.iloc[:,0].to_numpy(str)
                    predictions.extend([int(np.flatnonzero(names==v)[0]) for v in labels]);scores.append(result.probability_matrix.to_numpy())
                    score_names=result.probability_matrix.columns.to_numpy(str)
                save_case('celltypist',fold,seed,split,ids[split],y[ids[split]],np.asarray(predictions),groups,barcodes,names,np.concatenate(scores),score_names)
            write(case/'complete.json',dict(complete=True,package_version=importlib.metadata.version('celltypist'),code_sha256=sha(__file__)))

def run_singler():
    singler_numpy_compatibility()
    import singler
    x=normalize(sp.load_npz(EXPRESSION/'gse176078_brca_counts.npz').tocsr());genes=np.load(EXPRESSION/'gse176078_brca_genes.npy').astype(str)
    for fold in FOLDS:
        case=ROOT/f'singler_python/fold_{fold:02d}/seed_{SEEDS[0]}';case.mkdir(parents=True,exist_ok=True)
        if (case/'complete.json').exists():continue
        note('singler_reference_training',fold=fold)
        ids, y, names, groups, barcodes = fold_indices(fold)
        # Every training cell contributes. Official within-label vector-quantized
        # aggregation reduces a large single-cell reference without donor leakage.
        aggregate=singler.aggregate_reference(x[ids['train']].T.tocsc(),names[y[ids['train']]].tolist(),genes.tolist(),power=.5,num_threads=4)
        reference=singler.train_single(aggregate,aggregate.get_column_data()['label'],aggregate.get_row_names(),num_threads=4)
        markers=reference.marker_subset()
        write(case/'reference.json',dict(training_cells=len(ids['train']),aggregation='Official aggregate_reference power=.5, all training cells; no heldout cells',
            package='SingleR-inc/singler-py',version=importlib.metadata.version('singler'),markers=list(map(str,markers)),reference_fits_per_fold=1,code_sha256=sha(__file__),
            score_type='Spearman reference correlation scores, not calibrated probabilities',rejection='Native best-label output; no ad-hoc pruning threshold applied'))
        for split in ('valid','test'):
            predictions=[];scores=[]
            for start in range(0,len(ids[split]),2000):
                selected=ids[split][start:start+2000]
                result=singler.classify_single(x[selected].T.tocsc(),reference,num_threads=4)
                best=list(result['best']);predictions.extend([int(np.flatnonzero(names==str(v))[0]) for v in best])
                sf=result['scores'];sn=list(sf.get_column_names());scores.append(np.column_stack([np.asarray(sf[v]) for v in sn]))
            save_case('singler_python',fold,SEEDS[0],split,ids[split],y[ids[split]],np.asarray(predictions),groups,barcodes,names,np.concatenate(scores),sn)
        write(case/'complete.json',dict(complete=True,source_model_seed_not_applicable=True,reference_replicates=1,code_sha256=sha(__file__)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('pca', 'celltypist', 'singler'), required=True)
    parser.add_argument('--fold', type=int, choices=range(1, 6))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.fold:
        FOLDS = (args.fold,)
    if args.smoke:
        SMOKE, FOLDS, SEEDS = True, (args.fold or 1,), SEEDS[:1]
    ROOT.mkdir(parents=True, exist_ok=True)
    {'pca': run_pca, 'celltypist': run_celltypist, 'singler': run_singler}[args.method]()
    note('method_complete', method=args.method, folds=list(FOLDS))
