"""Locked external replication and reference-pool sensitivity, with patient-level inference."""
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
from gse_patient_programs_20261022 import project,rho,residual,align_export,transform_usage

MEDIUM=Path(__file__).resolve().parent
ROOT=BASE.parent/'gse176078_program_confirm_20261023'
PROGRAM=BASE.parent/'gse176078_patient_programs_20261022'
PREVIOUS=BASE.parent/'gse176078_program_classifiers_20261022'
EXTERNAL=BASE.parent/'gse176078_caf_mechanism_20261017/external/source'
METHODS=('mlp','gcn','gat','hgnn','fhnn','lighthgnn')
ARMS=('full',*METHODS,'pca')
RANDOM=20261023
PROTOCOL=dict(source_fold=1,seeds=list(SEEDS),locked_programs=[0,1],classifier_comparisons=list(METHODS),
    unsupervised_reference='PCA, reported and never an early-stop gate',
    main_endpoint='Within-patient measured-library-depth-adjusted B-score retrieval Spearman; both programs equal weight',
    primary_setting=dict(dimension='matched15',k=10,reference_pool=0),
    source_QC_bridge='Report depth-only, raw, and previous depth+mitochondrial score separately',
    external_QC='Measured library depth only: no mitochondrial genes in author matrices; never impute missing mitochondrial fraction as zero',
    secondary_settings=['native,k10','matched15,k30','native,k30'],
    reference_sensitivity_pools=[1,2,3,4,5],reference_seed=RANDOM,
    minimum_patient_cells=30,minimum_A_coverage=.7,minimum_B_coverage=1.,
    identity_policy='Exclude every patient ID appearing in more than one author object from primary inference; retain descriptive rows',
    bootstrap_patients=10000,signflip_MC_draws=100000,BH_family='Six classifier comparisons in external primary endpoint',
    neural_retraining=False,new_program_discovery=False,external_fitting=False,
    scope='GSE161529 is a separate source cohort, already explored in other analyses; not a pristine blinded validation')


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def event(stage,**kw):
    obj={'time':datetime.now(timezone.utc).isoformat(),'stage':stage,**kw}
    write(ROOT/'status.json',obj);print(json.dumps(obj),flush=True)


def canonical_patient(value):
    s=str(value).strip()
    return str(int(s)) if s.isdigit() else s


def object_folders():return [p for p in sorted(EXTERNAL.iterdir()) if (p/'manifest.json').exists()]


def identity_audit():
    records=[];allcells=[]
    for folder in object_folders():
        f=pd.read_csv(folder/'metadata.csv',dtype=str);m=f[f.label=='Malignant'].copy()
        m['patient']=m.patient.map(canonical_patient);m['object']=folder.name;allcells.append(m)
    frame=pd.concat(allcells,ignore_index=True)
    n=frame.groupby('patient').object.nunique();ambiguous=n[n>1].index.tolist()
    for (patient,folder),g in frame.groupby(['patient','object']):
        records.append(dict(patient=patient,object=folder,cells=len(g),eligible_primary=patient not in ambiguous,
            reason='ID occurs in multiple author objects; unresolved identity' if patient in ambiguous else ''))
    primary=frame[~frame.patient.isin(ambiguous)]
    audit=dict(ambiguous_ids=ambiguous,all_malignant_cells=len(frame),
        primary_malignant_cells=len(primary),primary_patients=primary.patient.nunique(),records=records)
    write(ROOT/'identity_audit.json',audit)
    return set(ambiguous)


def prepare_source():
    event('prepare_frozen_source_reference')
    out=ROOT/'source_inputs.npz'
    data=source_data()
    if out.exists():
        with np.load(out) as z:return data,{k:z[k].copy() for k in z.files}
    with np.load(PROGRAM/'source_program_usage.npz') as z:
        idx=z['idx'].copy();role=z['role'].copy();scores=z['B_scores'].copy();usage=z['usage'].copy()
    counts=sp.load_npz(BASE/'data/gse176078_brca_counts.npz').tocsr()[idx]
    totals=np.asarray(counts.sum(1)).ravel();expression=normalize(counts)
    with np.load(PCA_ROOT/'pca/fold_01.npz') as z:
        pc=(np.asarray(expression@z['components'].T)-z['gene_mean']@z['components'].T)/z['scale']
    meta=pd.read_csv(BASE.parent.parent/'data/biomed/Wu_etal_2021_BRCA_scRNASeq/metadata.csv',index_col=0).loc[data['barcodes'][idx]]
    # Keep old QC exact for the bridge; depth-only uses the same measured-count definition as external.
    legacy=np.column_stack([np.log1p(meta.nCount_RNA.to_numpy()),meta['percent.mito'].to_numpy()])
    values=dict(idx=idx,role=role,scores=scores,usage=usage,pca=pc,
        depth=np.log1p(totals)[:,None],legacy_QC=legacy,patients=data['patient_ids'][idx].astype(str))
    np.savez_compressed(out,**values)
    return data,values


def reference_pools(source):
    patients=source['patients'];train=source['role']==0;groups=np.unique(patients[train])
    with np.load(PROGRAM/'retrieval_split.npz') as z:original=z['reference_local'].copy()
    per=min(200,min(int((train&(patients==p)).sum()) for p in groups))
    pools=[original]
    for repeat in range(1,6):
        rng=np.random.default_rng(RANDOM+repeat)
        pools.append(np.concatenate([rng.choice(np.flatnonzero(train&(patients==p)),per,replace=False) for p in groups]))
    for ids in pools:
        assert np.all(train[ids]) and len(ids)==len(original)
        assert len(set(patients[ids]))==len(groups)
    np.savez_compressed(ROOT/'reference_pools.npz',**{f'pool_{j}':ids for j,ids in enumerate(pools)})
    return pools


def read_projection(arm,seed,mode):
    with np.load(PREVIOUS/f'retrieval/{arm}/seed_{seed}/{mode}/projection.npz') as z:
        return {k:z[k].copy() for k in z.files}


def measure(context,object_name,patients,queries,scores,usage,depth,legacy,points,source,
            arm,seed,pools,ambiguous):
    rows=[]
    settings=[('matched15',0),('native',0)]+[('matched15',j) for j in range(1,6)]
    for mode,pool in settings:
        parameters=read_projection(arm,seed,mode)
        ref=pools[pool];trainpoints=project(points['source'][ref],parameters)
        querypoints=project(points['query'],parameters)
        ks=[10,30] if pool==0 else [10]
        nn=NearestNeighbors(n_neighbors=max(ks)).fit(trainpoints)
        _,local=nn.kneighbors(querypoints);neighbors=ref[local]
        for k in ks:
            prediction=source['scores'][neighbors[:,:k]].mean(1)
            predicted_A=np.log1p(source['usage'][neighbors[:,:k]]).mean(1)
            for patient in np.unique(patients):
                take=patients==patient
                if take.sum()<30:continue
                eligible=(context!='external' or patient not in ambiguous)
                for program in (0,1):
                    raw=rho(prediction[take,program],scores[take,program])
                    adjusted=rho(residual(prediction[take,program],depth[take]),residual(scores[take,program],depth[take]))
                    old=(rho(residual(prediction[take,program],legacy[take]),residual(scores[take,program],legacy[take]))
                         if legacy is not None else None)
                    rows.append(dict(context=context,object=object_name,patient=patient,cells=int(take.sum()),
                        eligible_primary=eligible,arm=arm,seed=seed,dimension=mode,pool=pool,k=k,program=program,
                        rho_raw=raw,rho_depth=adjusted,rho_legacy_QC=old,
                        A_usage_rho=rho(predicted_A[take,program],np.log1p(usage[take,program]))))
    return rows


def external_targets(folder,source):
    frame=pd.read_csv(folder/'metadata.csv',dtype=str)
    ids=np.flatnonzero(frame.label.to_numpy()=='Malignant');frame=frame.iloc[ids].copy()
    genes=np.load(folder/'genes.npy').astype(str);lookup={g:i for i,g in enumerate(genes)}
    training_genes=np.load(BASE/'data/gse176078_brca_genes.npy').astype(str)
    train_lookup={g:i for i,g in enumerate(training_genes)}
    panels=json.loads((PROGRAM/'B_panels.json').read_text())
    with np.load(PROGRAM/'gene_preprocessing.npz') as z:
        mean=z['training_mean'].copy();sd=z['training_sd'].copy();a=z['A'].copy()
    with np.load(PROGRAM/'shared_program_dictionary.npz') as z:dictionary=z['H'].copy()
    counts=sp.load_npz(folder/'counts.npz').tocsr()[ids];totals=np.asarray(counts.sum(1)).ravel()
    assert (totals>0).all()
    log=normalize(counts)
    scores=np.empty((len(ids),2),np.float32)
    coverage=[]
    for panel in panels:
        wanted=panel['genes'];observed=[g for g in wanted if g in lookup]
        fraction=len(observed)/len(wanted);coverage.append(fraction)
        if fraction<1.:raise RuntimeError('Locked B panel is not fully measured; no silent gene replacement: '+folder.name)
        columns=[lookup[g] for g in wanted];training_columns=[train_lookup[g] for g in wanted]
        scores[:,panel['program']]=((log[:,columns].toarray()-mean[training_columns])/np.maximum(sd[training_columns],.1)).mean(1)
    measured=np.array([j for j,g in enumerate(training_genes[a]) if g in lookup])
    if len(measured)/len(a)<.7:raise RuntimeError('Inadequate A gene coverage')
    expression=np.zeros((len(ids),len(a)),np.float32)
    expression[:,measured]=log[:,[lookup[training_genes[a[j]]] for j in measured]].toarray()
    expression/=np.maximum(sd[a],.1)
    usage=transform_usage(expression,dictionary)
    destination=np.array([j for j,g in enumerate(training_genes) if g in lookup])
    src=np.array([lookup[training_genes[j]] for j in destination])
    with np.load(PCA_ROOT/'pca/fold_01.npz') as z:
        pc=(np.asarray(log[:,src]@z['components'][:,destination].T)-z['gene_mean']@z['components'].T)/z['scale']
    patient=frame.patient.map(canonical_patient).to_numpy(str);depth=np.log1p(totals)[:,None]
    write(ROOT/f'external_targets/{folder.name}_coverage.json',dict(B_coverage=coverage,
        A_coverage=len(measured)/len(a),mitochondrial_genes=int(np.char.startswith(genes,'MT-').sum()),
        depth_definition='sum of all measured author count genes, log1p',
        mitochondrial_fraction_imputed=False,missing_A_policy='Zero measured expression before frozen scaling',
        counts_manifest=json.loads((folder/'manifest.json').read_text())))
    np.savez_compressed(ROOT/f'external_targets/{folder.name}.npz',idx=ids,barcodes=frame.barcode.to_numpy(str),
        patients=patient,scores=scores,usage=usage,depth=depth)
    return ids,frame,patient,scores,usage,depth,pc


def paired_analysis(frame):
    f=frame[frame.eligible_primary].copy()
    # Both locked programs must be measurable; do not average a favorable subset.
    keys=['context','object','patient','arm','seed','dimension','pool','k']
    grouped=f.groupby(keys).agg(value=('rho_depth','mean'),valid_programs=('rho_depth','count')).reset_index()
    grouped.loc[grouped.valid_programs!=2,'value']=np.nan
    grouped.to_csv(ROOT/'patient_seed_pool_metrics.csv',index=False)
    descriptive=grouped.groupby(['context','object','dimension','pool','k','arm']).value.mean().reset_index()
    descriptive.to_csv(ROOT/'sensitivity_means.csv',index=False)
    primary=grouped[(grouped.dimension=='matched15')&(grouped.pool==0)&(grouped.k==10)]
    rows=[]
    for context,case in primary.groupby('context'):
        values=case.groupby(['patient','arm']).value.mean().unstack('arm')
        for comparator in (*METHODS,'pca'):
            pair=values[['full',comparator]].dropna();delta=(pair.full-pair[comparator]).to_numpy()
            if not len(delta):continue
            rng=np.random.default_rng(RANDOM)
            boot=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
            if len(delta)<=16:
                signs=np.array(list(itertools.product([-1,1],repeat=len(delta))))
                p=float((np.abs((signs*delta).mean(1))>=abs(delta.mean())-1e-12).mean());method='exact'
            else:
                signs=rng.choice([-1,1],size=(100000,len(delta)))
                p=float((1+(np.abs((signs*delta).mean(1))>=abs(delta.mean())-1e-12).sum())/100001);method='Monte Carlo, 100000 draws, +1 correction'
            low,high=np.quantile(boot,[.025,.975])
            rows.append(dict(context=context,comparator=comparator,category='reference' if comparator=='pca' else 'classifier',
                patients=len(delta),full_mean=float(pair.full.mean()),comparator_mean=float(pair[comparator].mean()),
                mean_delta=float(delta.mean()),positive_patient_fraction=float((delta>0).mean()),
                patient_bootstrap_95_low=float(low),patient_bootstrap_95_high=float(high),p=p,signflip_method=method))
    result=pd.DataFrame(rows);result['BH_q']=np.nan
    for context,case in result[result.category=='classifier'].groupby('context'):
        p=case.p.to_numpy();order=np.argsort(p)
        q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
        back=np.empty(len(p));back[order]=np.minimum(q,1);result.loc[case.index,'BH_q']=back
    result.to_csv(ROOT/'paired_primary_summary.csv',index=False)
    write(ROOT/'summary.json',dict(primary=result.astype(object).where(pd.notna(result),None).to_dict('records'),
        protocol=PROTOCOL,scope='External patient replication of frozen programs; same-source resampling is sensitivity only',
        no_stopping_for_PCA=True,no_model_or_program_selection=True))


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'complete.json').exists():print((ROOT/'complete.json').read_text());return
    write(ROOT/'protocol.json',PROTOCOL)
    ambiguous=identity_audit()
    locked=['shared_program_dictionary.npz','gene_preprocessing.npz','B_panels.json','source_program_usage.npz']
    write(ROOT/'locked_inputs.json',{n:sha(PROGRAM/n) for n in locked})
    for arm in ('full',*METHODS):
        for seed in SEEDS:
            if all((ROOT/f'external_representations/{arm}/seed_{seed}/{f.name}.npz').exists() for f in object_folders()):continue
            event('export_external',arm=arm,seed=seed)
            env=dict(os.environ)
            if arm!='fhnn':
                env['CUDA_VISIBLE_DEVICES']=''
            python=os.environ.get('CLHYP_LIGHT_PYTHON',sys.executable) if arm=='lighthgnn' else sys.executable
            log=ROOT/f'export_logs/{arm}_{seed}.log';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('w') as handle:
                subprocess.run([python,str(MEDIUM/'gse_program_confirm_export_20261023.py'),
                    '--method',arm,'--seed',str(seed)],cwd=MEDIUM,stdout=handle,stderr=subprocess.STDOUT,env=env,check=True)
    data,source=prepare_source();pools=reference_pools(source)
    root=ROOT/'cases';root.mkdir(exist_ok=True)
    for object_name in ['source_test',*[f.name for f in object_folders()]]:
        out=root/f'{object_name}.csv'
        if out.exists():continue
        event('evaluate_case',object=object_name)
        if object_name=='source_test':
            queries=np.flatnonzero(source['role']==2);ids=source['idx'][queries]
            patients=source['patients'][queries];scores=source['scores'][queries];usage=source['usage'][queries]
            depth=source['depth'][queries];legacy=source['legacy_QC'][queries];pc=source['pca'][queries]
            context='source_test';barcodes=data['barcodes'][ids]
        else:
            ids,frame,patients,scores,usage,depth,pc=external_targets(EXTERNAL/object_name,source)
            queries=None;legacy=None;context='external';barcodes=frame.barcode.to_numpy(str)
        rows=[]
        for seed in SEEDS:
            for arm in ARMS:
                source_path=source_export(seed) if arm=='full' else PREVIOUS/f'representations/{arm}/seed_{seed}.npz'
                sx=source['pca'] if arm=='pca' else align_export(source_path,source['idx'],source['role'])
                if context=='source_test':qx=sx[queries]
                elif arm=='pca':qx=pc
                else:
                    with np.load(ROOT/f'external_representations/{arm}/seed_{seed}/{object_name}.npz') as z:
                        assert np.array_equal(z['idx'],ids) and np.array_equal(z['barcodes'],barcodes)
                        qx=z['points'].astype(np.float64)
                rows.extend(measure(context,object_name,patients,queries,scores,usage,depth,legacy,
                    dict(source=sx,query=qx),source,arm,seed,pools,ambiguous))
        pd.DataFrame(rows).to_csv(out,index=False)
        bio=[]
        for patient in np.unique(patients):
            take=patients==patient
            for program in (0,1):
                bio.append(dict(patient=patient,program=program,cells=int(take.sum()),
                    raw_rho=rho(np.log1p(usage[take,program]),scores[take,program]),
                    depth_rho=rho(residual(np.log1p(usage[take,program]),depth[take]),residual(scores[take,program],depth[take])),
                    eligible_primary=patient not in ambiguous if context=='external' else True))
        pd.DataFrame(bio).to_csv(root/f'{object_name}_program_consistency.csv',index=False)
    files=[root/'source_test.csv',*[root/f'{f.name}.csv' for f in object_folders()]]
    metrics=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    metrics.to_csv(ROOT/'all_patient_program_metrics.csv',index=False);paired_analysis(metrics)
    write(ROOT/'complete.json',dict(time=datetime.now(timezone.utc).isoformat(),status='complete',
        source_fold=1,seeds=list(SEEDS),external_cohort='GSE161529',classifiers=list(METHODS),
        retraining=False,new_program_discovery=False,PCA_gate=False,spatial_mapping=False))
    event('complete',result=str(ROOT/'summary.json'))


def check():
    ROOT.mkdir(parents=True,exist_ok=True);identity_audit()
    panels=json.loads((PROGRAM/'B_panels.json').read_text())
    coverage=[]
    with np.load(PROGRAM/'shared_program_dictionary.npz') as z:A=z['A_genes'].astype(str)
    for folder in object_folders():
        genes=np.load(folder/'genes.npy').astype(str);f=pd.read_csv(folder/'metadata.csv',dtype=str)
        assert len(set(genes))==len(genes)
        cov=[float(np.isin(panel['genes'],genes).mean()) for panel in panels]
        assert min(cov)==1. and np.isin(A,genes).mean()>=.7
        for sample in f['sample'].unique():
            assert (BASE.parent/f'gse176078_caf_final_compare_20261018/external_inputs/{folder.name}/fold_01_{sample}_edges.npy').exists()
        coverage.append(dict(object=folder.name,A=float(np.isin(A,genes).mean()),B=cov,
            mitochondrial_genes=int(np.char.startswith(genes,'MT-').sum())))
    for seed in SEEDS:
        for arm in ARMS:
            for mode in ('matched15','native'):
                parameters=read_projection(arm,seed,mode)
                assert parameters['components'].shape[1]==15 if mode=='matched15' else True
    with np.load(PROGRAM/'source_program_usage.npz') as z:
        train=z['role']==0;patients=source_data()['patient_ids'][z['idx']]
        with np.load(PROGRAM/'retrieval_split.npz') as d:assert np.all(train[d['reference_local']])
        assert not set(patients[train])&set(patients[z['role']==2])
    # Depth residualization and within-patient matching must work without inventing missing mitochondrial values.
    rng=np.random.default_rng(23);q=rng.normal(size=(100,1));y=2*q[:,0]+rng.normal(size=100)
    assert abs(np.corrcoef(residual(y,q),q[:,0])[0,1])<1e-8
    write(ROOT/'startup_check.json',dict(passed=True,coverage=coverage,reference_training_only=True,
        external_fit=False,mitochondrial_missing_not_zero=True,protocol=PROTOCOL))
    print(json.dumps(dict(preflight_passed=True,coverage=coverage)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    with threadpool_limits(limits=4):
        if parser.parse_args().check:check()
        else:
            try:run()
            except Exception as exc:
                event('failed',error_type=type(exc).__name__,message=str(exc));raise
