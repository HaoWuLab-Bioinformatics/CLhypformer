"""Compare frozen final classifier representation, PCA and input SVD fairly."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.linear_model import Ridge
from scipy.stats import wilcoxon
import gse_caf_pca_compare_20261018 as common
from gse_caf_analysis_20261017 import BASE,SEEDS,CAF_SIGNATURES,write,sha,bh
from gse_caf_continuous_20261017 import ALPHAS,fit_scaler,equal_donor_weights,rho,read_markers
from gse_caf_mechanism_runtime_20261017 import load_full,project_external
from gse_patient_state_graph import patient_local_edges

ROOT=BASE.parent/'gse176078_caf_final_compare_20261018'
OLD=common.ROOT
ARMS=('pca','svd','full')
MODES=('linear','rbf')

def note(stage,**kwargs):
    from datetime import datetime,timezone
    record=dict(stage=stage,time=datetime.now(timezone.utc).isoformat(),**kwargs)
    write(ROOT/'status.json',record);print(json.dumps(record),flush=True)

def final_embedding(model,x,prior,groups,edges):
    with torch.no_grad():
        fused=model.state_module(torch.from_numpy(x),torch.from_numpy(prior),torch.from_numpy(groups).long())
        logits,z=model._forward_task(fused,torch.from_numpy(edges).long(),return_embedding=True)
        size=model.decode_trans.in_features
        weight=torch.cat([model.decode_trans.weight,model.decode_graph.weight],dim=1)
        bias=(1-model.graph_weight)*model.decode_trans.bias+model.graph_weight*model.decode_graph.bias
        replay=z@weight.T+bias
        error=float((replay-logits).abs().max())
        if not torch.allclose(replay,logits,atol=2e-5,rtol=2e-5):raise ValueError(('Classifier replay failed',error))
        if z.shape[1]!=2*size:raise ValueError('Unexpected final representation')
    return z.numpy().astype(np.float64),logits.softmax(-1).numpy(),error

def source_export(fold,seed):
    output=ROOT/f'representations/fold_{fold:02d}/seed_{seed}'
    if (output/'complete.json').exists():return output
    note('export_final_source',fold=fold,seed=seed)
    output.mkdir(parents=True,exist_ok=True);model=load_full(fold,seed)
    with np.load(BASE/f'data/fold_{fold:02d}/biomed/gse176078_brca.npz') as d:
        x=d['features'].copy();patient=d['patient_ids'].astype(str);edges=d['edge_index'].copy();labels=d['labels'].copy();names=d['class_names'].astype(str)
    records=[];maximum=0.;head_error=0.
    for saved in sorted((BASE/f'runs/state_hyper_cl/fold_{fold:02d}/seed_{seed}/state_representations').glob('*.npz')):
        if saved.name!='train.npz' and not saved.name.startswith('heldout_'):continue
        with np.load(saved) as d:
            ids=d['global_idx'].copy();prior=d['activity_prior'].copy();reference=d['probability'].copy()
        # Check heldout files really are test patients, never validation labels.
        with np.load(BASE/f'data/fold_{fold:02d}/biomed/gse176078_brca.npz') as d:
            allowed=d['train_idx'] if saved.name=='train.npz' else d['test_idx']
        if not np.isin(ids,allowed).all():continue
        mapping=np.full(len(x),-1,dtype=np.int64);mapping[ids]=np.arange(len(ids))
        keep=(mapping[edges[0]]>=0)&(mapping[edges[1]]>=0)
        _,codes=np.unique(patient[ids],return_inverse=True)
        z,p,error=final_embedding(model,x[ids],prior,codes,mapping[edges[:,keep]])
        drift=float(np.abs(p-reference).max());maximum=max(maximum,drift);head_error=max(head_error,error)
        if drift>2e-4:raise ValueError(('Saved prediction differs',saved.name,drift))
        caf=names[labels[ids]]=='CAFs'
        records.append((ids[caf],z[caf],np.full(caf.sum(),saved.name=='train.npz')))
    idx=np.concatenate([r[0] for r in records]);points=np.concatenate([r[1] for r in records]);train=np.concatenate([r[2] for r in records])
    if len(np.unique(idx))!=len(idx):raise ValueError('Duplicate source CAF exports')
    np.savez_compressed(output/'source.npz',idx=idx,points=points,train=train)
    write(output/'complete.json',dict(source_probability_max_error=maximum,classifier_head_max_error=head_error,
        dimensions=points.shape[1],checkpoint_sha256=sha(BASE/f'runs/state_hyper_cl/fold_{fold:02d}/seed_{seed}/checkpoint.pt'),
        representation='Weighted concatenation of logmap0+LayerNorm Transformer branch and ordinary graph branch before two affine classifier heads',
        space='Euclidean direct-product representation, not Lorentz coordinates'))
    return output

def fit_linear(x,y,groups,alpha):
    xm,xs=fit_scaler(x,groups);ym,ys=fit_scaler(y,groups)
    model=Ridge(alpha=alpha,solver='cholesky').fit((x-xm)/xs,(y-ym)/ys,sample_weight=equal_donor_weights(groups))
    return dict(xm=xm,xs=xs,ym=ym,ys=ys,coef=model.coef_,intercept=model.intercept_)

def fit_readout(x,y,groups,mode,seed,alpha):
    if mode=='linear':return fit_linear(x,y,groups,alpha)
    # Scale each embedding from training donors before equal-budget RBF features.
    xm,xs=fit_scaler(x,groups)
    read=common.fit((x-xm)/xs,np.empty((len(x),0)),y,groups,None,seed,alpha)
    read['input_mean']=xm;read['input_sd']=xs
    return read

def predict(model,x,mode):
    if mode=='linear':return (((x-model['xm'])/model['xs'])@model['coef'].T+model['intercept'])*model['ys']+model['ym']
    xx=(x-model['input_mean'])/model['input_sd']
    return common.predict(model,xx,np.empty((len(x),0)))

def select(x,y,groups,mode,seed):
    losses={a:[] for a in ALPHAS}
    for j,group in enumerate(np.unique(groups)):
        valid=groups==group;train=~valid
        for a in ALPHAS:
            read=fit_readout(x[train],y[train],groups[train],mode,seed+300+j,a)
            p=predict(read,x[valid],mode)
            losses[a].append(float(np.mean(((p-y[valid])/read['ys'])**2)))
    a=min(ALPHAS,key=lambda a:(np.mean(losses[a]),-a))
    return fit_readout(x,y,groups,mode,seed,a),a,{str(a):float(np.mean(v)) for a,v in losses.items()}

def save_prediction(path,barcodes,patients,pred,truth):
    result=pd.DataFrame(dict(barcode=barcodes,patient=patients))
    for j,name in enumerate(CAF_SIGNATURES):result['score_'+name]=pred[:,j];result['expression_'+name+'_B']=truth[:,j]
    result.to_csv(path,index=False)

def run():
    torch.set_num_threads(4);ROOT.mkdir(parents=True,exist_ok=True)
    write(ROOT/'protocol.json',dict(arms=ARMS,modes=MODES,primary='full - pca, linear readout; source/external x ECM/contractile/interferon, six BH tests',
        source='Original five patient folds x two seeds; train/readout uses training CAF only',external='Previously explored GSE161529; no external fitting',
        representation='Final classifier input; no appended memberships, programs or class probabilities',
        geometry='Transformer logmap0 in frozen model, concatenate weighted ordinary-graph representation; both readouts Euclidean',
        preprocessing='Reuse fixed outer-train PCA/SVD. Only readout is inner donor-crossfit; no fully nested representation fitting claim',
        dimension='PCA/SVD 50; model native dimension reported in export audit, not dimension matched',
        uncertainty='Average cell predictions first; paired patient bootstrap 5000; >=30 CAF; raw and QC-controlled correlations',
        marker_limitation='B genes excluded from readout targets but may be upstream expression inputs',
        sha256={p:sha(Path(__file__).parent/p) for p in [Path(__file__).name,'hypformer.py','gse_caf_mechanism_runtime_20261017.py','gse_caf_pca_compare_20261018.py']}))
    genes=np.load(BASE/'data/gse176078_brca_genes.npy').astype(str)
    normalized=common.normalize(sp.load_npz(BASE/'data/gse176078_brca_counts.npz').tocsr());panels,_=read_markers(normalized,genes)
    for fold in range(1,6):
        with np.load(BASE/f'data/fold_{fold:02d}/biomed/gse176078_brca.npz') as d:
            svd=d['features'].copy();patient=d['patient_ids'].astype(str);barcodes=d['barcodes'].astype(str)
        with np.load(OLD/f'pca/fold_{fold:02d}.npz') as d:pca={k:d[k].copy() for k in d.files}
        for seed in SEEDS:
            case=ROOT/f'cases/fold_{fold:02d}/seed_{seed}';case.mkdir(parents=True,exist_ok=True)
            if (case/'complete.json').exists():continue
            rep=source_export(fold,seed)
            with np.load(rep/'source.npz') as d:idx=d['idx'].copy();z=d['points'].copy();training=d['train'].copy()
            xx=dict(pca=common.project(normalized[idx],pca),svd=svd[idx],full=z);groups=patient[idx]
            if set(groups[training])&set(groups[~training]):raise ValueError('Patient leakage')
            reads={}
            for mode in MODES:
                for arm in ARMS:
                    note('fit_functional_readout',fold=fold,seed=seed,mode=mode,arm=arm)
                    read,a,loss=select(xx[arm][training],panels['A'][idx[training]],groups[training],mode,seed+fold)
                    reads[mode,arm]=read
                    np.savez_compressed(case/f'{mode}_{arm}_readout.npz',**{k:v for k,v in read.items() if k!='k'})
                    write(case/f'{mode}_{arm}_selection.json',dict(alpha=a,inner_losses=loss))
                    save_prediction(case/f'{mode}_{arm}_source.csv',barcodes[idx[~training]],groups[~training],predict(read,xx[arm][~training],mode),panels['B'][idx[~training]])
            model=load_full(fold,seed)
            truth=pd.read_csv(common.MECHANISM/'external/CAF_frozen_ensemble_scores.csv',dtype={'barcode':str,'patient':str}).set_index('barcode')
            for folder,counts,external_genes,frame,caf,aligned,_ in common.external_inputs():
                note('export_final_external',fold=fold,seed=seed,source=folder.name)
                cache=ROOT/f'external_inputs/{folder.name}/fold_{fold:02d}.npz';cache.parent.mkdir(parents=True,exist_ok=True)
                if cache.exists():
                    with np.load(cache) as d:features=d['features'].copy();prior=d['prior'].copy()
                else:
                    features,prior,coverage=project_external(counts,external_genes,fold)
                    np.savez_compressed(cache,features=features,prior=prior)
                    write(cache.with_suffix('.json'),coverage)
                full=np.full((len(caf),z.shape[1]),np.nan);pos=np.full(len(frame),-1,dtype=int);pos[caf]=np.arange(len(caf));audits=[]
                for sample,rows in frame.groupby('sample'):
                    ids=rows.index.to_numpy();mask=rows.label.to_numpy()=='Fibroblasts'
                    if not mask.any():continue
                    graph_path=cache.parent/f'fold_{fold:02d}_{sample}_edges.npy'
                    if graph_path.exists():edges=np.load(graph_path)
                    else:
                        edges,_=patient_local_edges(features[ids],np.zeros(len(ids),dtype=int),k=5);np.save(graph_path,edges)
                    points,_,error=final_embedding(model,features[ids],prior[ids],np.zeros(len(ids),dtype=np.int64),edges)
                    full[pos[ids[mask]]]=points[mask];audits.append(dict(sample=sample,classifier_head_max_error=error,cells=len(ids)))
                if not np.isfinite(full).all():raise ValueError('Missing external CAF final representations')
                np.savez_compressed(rep/f'external_{folder.name}.npz',barcodes=frame.iloc[caf].barcode.to_numpy(str),points=full)
                write(rep/f'external_{folder.name}_audit.json',audits)
                ex=dict(pca=common.project(aligned,pca),svd=features[caf],full=full)
                b=frame.iloc[caf].barcode.to_numpy(str);patients=frame.iloc[caf].patient.to_numpy(str)
                target=np.column_stack([truth.loc[b,'expression_'+name+'_B'].to_numpy() for name in CAF_SIGNATURES])
                for mode in MODES:
                    for arm in ARMS:save_prediction(case/f'{mode}_{arm}_external_{folder.name}.csv',b,patients,predict(reads[mode,arm],ex[arm],mode),target)
            write(case/'complete.json',dict(complete=True))
    summarize()

def summarize():
    # Reuse exactly the patient/covariate audit rules of the preceding comparison.
    meta=pd.read_csv(Path(__file__).parent.parent/'data/biomed/Wu_etal_2021_BRCA_scRNASeq/metadata.csv',index_col=0)
    qc_source=pd.DataFrame(dict(barcode=meta.index.astype(str),q1=np.log1p(meta.nCount_RNA.to_numpy()),q2=np.log1p(meta.nFeature_RNA.to_numpy()),q3=meta['percent.mito'].to_numpy())).set_index('barcode')
    qlist=[]
    for folder in sorted((common.MECHANISM/'external/source').iterdir()):
        if not (folder/'manifest.json').exists():continue
        c=sp.load_npz(folder/'counts.npz').tocsr();g=np.load(folder/'genes.npy').astype(str);m=pd.read_csv(folder/'metadata.csv',dtype=str)
        total=np.asarray(c.sum(1)).ravel();mito=np.asarray(c[:,np.char.startswith(np.char.upper(g),'MT-')].sum(1)).ravel()
        qlist.append(pd.DataFrame(dict(barcode=m.barcode,q1=np.log1p(total),q2=np.log1p(c.getnnz(1)),q3=100*mito/np.maximum(total,1))).set_index('barcode'))
    qc_external=pd.concat(qlist);rows=[]
    for mode in MODES:
        for cohort in ('source','external'):
            for arm in ARMS:
                files=list((ROOT/'cases').glob(f'fold_*/seed_*/{mode}_{arm}_{cohort}*.csv'))
                d=pd.concat([pd.read_csv(p,dtype={'barcode':str,'patient':str}) for p in files]).groupby(['barcode','patient'],as_index=False).mean(numeric_only=True)
                d.to_csv(ROOT/f'{mode}_{arm}_{cohort}_ensemble.csv',index=False)
                q=(qc_source if cohort=='source' else qc_external).loc[d.barcode].to_numpy()
                for patient,ids in d.groupby('patient').groups.items():
                    ids=np.asarray(list(ids));t=d.loc[ids]
                    if len(t)<30:continue
                    for name in CAF_SIGNATURES:
                        x=t['score_'+name].to_numpy();y=t['expression_'+name+'_B'].to_numpy();valid=np.isfinite(x)&np.isfinite(y)
                        if valid.sum()<30:continue
                        rows.append(dict(mode=mode,cohort=cohort,arm=arm,patient=patient,program=name,cells=int(valid.sum()),rho=rho(x[valid],y[valid]),qc_rho=common.partial_rho(x[valid],y[valid],q[ids][valid])))
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'patient_correlations.csv',index=False);results=[];rng=np.random.default_rng(181018)
    for mode in MODES:
        for cohort in ('source','external'):
            for metric in ('rho','qc_rho'):
                for comparator in ('pca','svd'):
                    for name in CAF_SIGNATURES:
                        t=frame[(frame['mode']==mode)&(frame.cohort==cohort)&(frame.program==name)].pivot(index='patient',columns='arm',values=metric)[['full',comparator]].dropna()
                        delta=(t.full-t[comparator]).to_numpy()
                        if not len(delta):continue
                        ci=np.quantile(rng.choice(delta,(5000,len(delta)),replace=True).mean(1),[.025,.975])
                        results.append(dict(mode=mode,cohort=cohort,metric=metric,comparator=comparator,program=name,patients=len(delta),mean_delta=float(delta.mean()),CI95=ci.tolist(),improved_fraction=float((delta>0).mean()),p=float(wilcoxon(delta,method='approx').pvalue) if np.any(delta!=0) else 1.,primary=mode=='linear' and metric=='rho' and comparator=='pca' and name!='inflammatory'))
    for primary in (True,False):
        family=[r for r in results if r['primary']==primary]
        for r,q in zip(family,bh([r['p'] for r in family])):r['BH_q']=float(q)
    write(ROOT/'paired_summary.json',results);note('complete',cases=10,arms=ARMS,modes=MODES)

def check():
    torch.set_num_threads(4)
    model=load_full(1,SEEDS[0])
    with np.load(BASE/'data/fold_01/biomed/gse176078_brca.npz') as d:
        ids=d['test_idx'];patients=d['patient_ids'].astype(str);ids=ids[patients[ids]==np.unique(patients[ids])[0]]
        x=d['features'][ids];edges=d['edge_index'];mapping=np.full(len(patients),-1,dtype=np.int64);mapping[ids]=np.arange(len(ids));keep=(mapping[edges[0]]>=0)&(mapping[edges[1]]>=0);edges=mapping[edges[:,keep]]
    with np.load(BASE/f'runs/state_hyper_cl/fold_01/seed_{SEEDS[0]}/state_representations/heldout_{patients[ids][0]}.npz') as d:
        assert np.array_equal(ids,d['global_idx']);prior=d['activity_prior'].copy();ref=d['probability'].copy()
    z,p,error=final_embedding(model,x,prior,np.zeros(len(ids),dtype=np.int64),edges)
    drift=float(np.abs(p-ref).max());assert drift<2e-4
    rng=np.random.default_rng(18);xx=rng.normal(size=(90,8));y=xx[:,:4];groups=np.repeat(['a','b','c'],30)
    for mode in MODES:
        read=fit_readout(xx,y,groups,mode,18,1.);assert np.isfinite(predict(read,xx,mode)).all()
    record=dict(real_classifier_replay_max_error=error,saved_probability_max_error=drift,final_dimension=z.shape[1],source_cells=len(ids),readouts_checked=MODES)
    write(ROOT/'startup_check.json',record);print(json.dumps(record),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:check()
    else:run()
