"""Horizontal classifier comparison. PCA is retained as an unsupervised reference."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='4'
import argparse
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from gse_caf_analysis_20261017 import BASE,sha
from gse_caf_pca_compare_20261018 import normalize,ROOT as PCA_ROOT
from gse_tumor_state_runtime_20261022 import source_data,source_export,SEEDS
from gse_patient_programs_20261022 import fit_coordinates,project,rho,residual,align_export

MEDIUM=Path(__file__).resolve().parent
ROOT=BASE.parent/'gse176078_program_classifiers_20261022'
PREVIOUS=BASE.parent/'gse176078_patient_programs_20261022'
OLD=BASE.parent/'gse176078_caf_confirm_20261022'
METHODS=('mlp','gcn','gat','hgnn','fhnn','lighthgnn')
RANDOM=20261022
PROTOCOL=dict(fold=1,seeds=list(SEEDS),primary_comparators=list(METHODS),
    secondary_reference='PCA; never used as a continuation/selection gate',
    main_dimension=15,dimension_reason='Common dimension includes the 15D FHNN tangent output',
    k=[10,30],primary_k=10,programs=[0,1],classifier_retraining=False,
    primary_metric='Patient-equal mean over both locked programs of QC-adjusted B-score retrieval Spearman',
    roles={'validation':'Already explored development patients','test':'Within-cohort held-out patients; not an independent new cohort'},
    policy='Run and report every classifier, both programs, native and matched15; no PCA gate',
    revision='After seeing the previous PCA comparison, user explicitly changed the primary scientific comparison; not an original preregistration')


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def event(stage,**kw):
    obj={'time':datetime.now(timezone.utc).isoformat(),'stage':stage,**kw}
    write(ROOT/'status.json',obj);print(json.dumps(obj),flush=True)


def summary(frame):
    # No program or model selection based on these results.
    aggregate=frame.groupby(['role','dimension','k','arm','seed','patient']).B_qc_rho.mean().reset_index()
    aggregate.to_csv(ROOT/'patient_seed_aggregate.csv',index=False)
    rows=[]
    for (role,dimension,k),part in aggregate.groupby(['role','dimension','k']):
        values=part.groupby(['patient','arm']).B_qc_rho.mean().unstack('arm')
        for comparator in (*METHODS,'pca'):
            if comparator not in values or 'full' not in values:continue
            pair=values[['full',comparator]].dropna();delta=(pair.full-pair[comparator]).to_numpy()
            if not len(delta):continue
            rng=np.random.default_rng(RANDOM)
            interval=np.quantile(delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1),[.025,.975])
            signs=np.array(list(itertools.product([-1,1],repeat=len(delta))))
            p=float((np.abs((signs*delta).mean(1))>=abs(delta.mean())-1e-12).mean())
            rows.append(dict(role=role,dimension=dimension,k=int(k),comparator=comparator,
                category='unsupervised_reference' if comparator=='pca' else 'classifier',patients=len(pair),
                full_mean=float(pair.full.mean()),comparator_mean=float(pair[comparator].mean()),
                mean_delta=float(delta.mean()),positive_patient_fraction=float((delta>0).mean()),
                patient_bootstrap_95_low=float(interval[0]),patient_bootstrap_95_high=float(interval[1]),
                exact_patient_signflip_p=p))
    out=pd.DataFrame(rows)
    if len(out):
        out['BH_q']=np.nan
        for (role,dimension,k),g in out[out.category=='classifier'].groupby(['role','dimension','k']):
            p=g.exact_patient_signflip_p.to_numpy();order=np.argsort(p)
            ordered=p[order]*len(p)/np.arange(1,len(p)+1)
            adjusted=np.minimum.accumulate(ordered[::-1])[::-1]
            q=np.empty(len(p));q[order]=np.minimum(adjusted,1)
            out.loc[g.index,'BH_q']=q
        out.to_csv(ROOT/'paired_summary.csv',index=False)
    primary=out[(out.dimension=='matched15')&(out.k==10)] if len(out) else out
    write(ROOT/'summary.json',dict(primary=primary.astype(object).where(pd.notna(primary),None).to_dict('records'),
        note='PCA retained, no advantage gate; patient count rather than cells/seeds determines replication',
        all_classifier_results_reported=True,program_selection=False))


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'complete.json').exists():print((ROOT/'complete.json').read_text());return
    write(ROOT/'protocol.json',PROTOCOL)
    locked=['gene_preprocessing.npz','shared_program_dictionary.npz','source_program_usage.npz',
            'B_panels.json','retrieval_split.npz']
    write(ROOT/'locked_input_hashes.json',{name:sha(PREVIOUS/name) for name in locked})
    failures=[]
    for method in METHODS:
        for seed in SEEDS:
            output=ROOT/f'representations/{method}/seed_{seed}.npz'
            if output.exists():continue
            event('export_classifier',method=method,seed=seed)
            python=os.environ.get('CLHYP_LIGHT_PYTHON',sys.executable) if method=='lighthgnn' else sys.executable
            log=ROOT/f'export_logs/{method}_{seed}.log';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('w') as handle:
                process=subprocess.run([python,str(MEDIUM/'gse_program_classifier_export_20261022.py'),
                    '--method',method,'--seed',str(seed)],cwd=MEDIUM,stdout=handle,stderr=subprocess.STDOUT)
            if process.returncode:
                failures.append(dict(method=method,seed=seed,log=str(log)))
    write(ROOT/'export_failures.json',failures)
    if failures:
        raise RuntimeError('Comparator export failed; all failures retained, not silently excluded: '+str(failures))
    event('retrieval_all_classifiers')
    data=source_data()
    with np.load(PREVIOUS/'source_program_usage.npz') as z:
        idx=z['idx'].copy();role=z['role'].copy();scores=z['B_scores'].copy();usage=z['usage'].copy()
    patients=data['patient_ids'][idx].astype(str);train=role==0
    with np.load(PREVIOUS/'retrieval_split.npz') as z:reference=z['reference_local'].copy()
    assert np.all(role[reference]==0)
    expression=normalize(sp.load_npz(BASE/'data/gse176078_brca_counts.npz').tocsr()[idx])
    with np.load(PCA_ROOT/'pca/fold_01.npz') as z:
        pc=(np.asarray(expression@z['components'].T)-z['gene_mean']@z['components'].T)/z['scale']
    del expression
    meta=pd.read_csv(BASE.parent.parent/'data/biomed/Wu_etal_2021_BRCA_scRNASeq/metadata.csv',index_col=0).loc[data['barcodes'][idx]]
    qc=np.column_stack([np.log1p(meta.nCount_RNA.to_numpy()),meta['percent.mito'].to_numpy()])
    rows=[]
    for seed in SEEDS:
        points={'full':align_export(source_export(seed),idx,role),'pca':pc}
        points.update({method:align_export(ROOT/f'representations/{method}/seed_{seed}.npz',idx,role) for method in METHODS})
        assert min(x.shape[1] for x in points.values())>=15
        for arm,x in points.items():
            for dimension,size in [('matched15',15),('native',None)]:
                parameters=fit_coordinates(x[train],patients[train],size);z=project(x,parameters)
                out=ROOT/f'retrieval/{arm}/seed_{seed}/{dimension}';out.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(out/'projection.npz',**parameters)
                nn=NearestNeighbors(n_neighbors=30).fit(z[reference])
                for label,code in [('validation',1),('test',2)]:
                    queries=np.flatnonzero(role==code)
                    distances,local=nn.kneighbors(z[queries]);neighbors=reference[local]
                    np.savez_compressed(out/f'{label}_neighbors.npz',query_source_idx=idx[queries],
                        neighbor_source_idx=idx[neighbors],distances=distances)
                    for k in (10,30):
                        predicted=scores[neighbors[:,:k]].mean(1)
                        predicted_A=np.log1p(usage[neighbors[:,:k]]).mean(1)
                        for patient in np.unique(patients[queries]):
                            take=patients[queries]==patient;q=queries[take]
                            if len(q)<30:continue
                            for program in (0,1):
                                rows.append(dict(role=label,arm=arm,seed=seed,dimension=dimension,k=k,
                                    patient=patient,program=program,cells=len(q),B_rho=rho(predicted[take,program],scores[q,program]),
                                    B_qc_rho=rho(residual(predicted[take,program],qc[q]),residual(scores[q,program],qc[q])),
                                    A_usage_rho=rho(predicted_A[take,program],np.log1p(usage[q,program]))))
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'patient_program_metrics.csv',index=False)
    summary(frame)
    write(ROOT/'complete.json',dict(time=datetime.now(timezone.utc).isoformat(),status='complete',
        classifiers=list(METHODS),PCA_reported=True,stopped_for_PCA=False,
        classifier_retraining=False,external_evaluation=False,spatial_mapping=False,
        scope='Fold1 validation development + held-out test, two frozen model seeds; all outcomes retained'))
    event('complete',result=str(ROOT/'summary.json'))


def check():
    ROOT.mkdir(parents=True,exist_ok=True)
    gates=json.loads((PREVIOUS/'program_validation_gates.json').read_text())
    assert [g['program'] for g in gates if g['passed']]==[0,1]
    for method in METHODS:
        for seed in SEEDS:
            record=json.loads((OLD/f'selected/{method}/fold_01/seed_{seed}/selection.json').read_text())
            assert Path(record['checkpoint']).exists()
    with np.load(PREVIOUS/'retrieval_split.npz') as z:
        ref=z['reference_source_idx'].copy()
    data=source_data();assert np.isin(ref,data['train_idx']).all()
    assert not set(data['patient_ids'][data['train_idx']])&set(data['patient_ids'][data['test_idx']])
    rng=np.random.default_rng(22);x=rng.normal(size=(90,20));g=np.repeat(['a','b','c'],30)
    pars=fit_coordinates(x,g,15);assert project(x,pars).shape==(90,15)
    # Null correlation stays missing instead of becoming a favorable number.
    assert rho(np.ones(30),np.arange(30)) is None
    write(ROOT/'startup_check.json',dict(passed=True,all_six_selected_checkpoints_exist=True,
        reference_training_only=True,common15_projection=True,PCA_not_a_gate=True,protocol=PROTOCOL))
    print(json.dumps(dict(preflight_passed=True,classifiers=list(METHODS))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    with threadpool_limits(limits=4):
        if parser.parse_args().check:check()
        else:
            try:run()
            except Exception as exc:
                event('failed',error_type=type(exc).__name__,message=str(exc));raise
