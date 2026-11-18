"""Frozen-representation comparison; patient-level inference, no classifier refit."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, svds
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, wilcoxon
from sklearn.linear_model import Ridge
from gse_caf_analysis_20261017 import BASE, ROOT as PREVIOUS, SEEDS, CAF_SIGNATURES, write, sha, bh
from gse_caf_continuous_20261017 import ALPHAS, dist, fit_scaler, equal_donor_weights, read_markers, rho

ROOT=BASE.parent/'gse176078_caf_pca_compare_20261018'
MECHANISM=BASE.parent/'gse176078_caf_mechanism_20261017'
ARMS=('pca','pca_prior','svd_prior','model')

def note(stage,**kwargs):
    from datetime import datetime, timezone
    value=dict(stage=stage,time=datetime.now(timezone.utc).isoformat(),**kwargs)
    write(ROOT/'status.json',value);print(json.dumps(value),flush=True)

def normalize(counts):
    x=counts.multiply((1e4/np.maximum(np.asarray(counts.sum(1)).ravel(),1))[:,None]).tocsr()
    x.data=np.log1p(x.data)
    return x

def centered_pca(x,train,fold):
    path=ROOT/f'pca/fold_{fold:02d}.npz'
    if path.exists():
        with np.load(path) as d:return {k:d[k].copy() for k in d.files}
    tx=x[train].astype(np.float64);mean=np.asarray(tx.mean(0)).ravel();n,g=tx.shape
    # Explicit centering through an operator avoids a dense cell-by-gene matrix.
    op=LinearOperator((n,g),dtype=np.float64,
        matvec=lambda v:tx@v-np.full(n,mean@v),
        rmatvec=lambda v:tx.T@v-mean*v.sum(),
        matmat=lambda v:tx@v-np.ones((n,1))@(mean@v)[None,:],
        rmatmat=lambda v:tx.T@v-mean[:,None]*v.sum(0)[None,:])
    _,s,vt=svds(op,k=50,which='LM',tol=1e-6,random_state=168023018+fold)
    order=np.argsort(s)[::-1];components=vt[order]
    projected=np.asarray(tx@components.T)-mean@components.T
    scale=np.maximum(projected.std(0),1e-6)
    result=dict(components=components,gene_mean=mean,scale=scale,singular_values=s[order])
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**result)
    return result

def project(x,pca):return (np.asarray(x@pca['components'].T)-pca['gene_mean']@pca['components'].T)/pca['scale']

def distances(x,c,k):return cdist(x,c) if k is None else dist(x,c,k)

def fit(x,h,y,groups,k,seed,alpha):
    rng=np.random.default_rng(seed);ids=[]
    for group in np.unique(groups):
        rows=np.flatnonzero(groups==group);ids.extend(rng.choice(rows,min(4,len(rows)),replace=False))
    centers=x[ids];d=distances(x,centers,k);bandwidth=max(float(np.median(d)),.05)
    design=np.column_stack([np.exp(-.5*(d/bandwidth)**2),h])
    xm,xs=fit_scaler(design,groups);ym,ys=fit_scaler(y,groups)
    model=Ridge(alpha=alpha,solver='cholesky').fit((design-xm)/xs,(y-ym)/ys,sample_weight=equal_donor_weights(groups))
    return dict(centers=centers,bandwidth=bandwidth,k=k,xm=xm,xs=xs,ym=ym,ys=ys,coef=model.coef_,intercept=model.intercept_)

def predict(model,x,h):
    d=distances(x,model['centers'],model['k'])
    design=np.column_stack([np.exp(-.5*(d/model['bandwidth'])**2),h])
    return (((design-model['xm'])/model['xs'])@model['coef'].T+model['intercept'])*model['ys']+model['ym']

def select(x,h,y,groups,k,seed):
    errors={a:[] for a in ALPHAS}
    for j,group in enumerate(np.unique(groups)):
        valid=groups==group;train=~valid
        for a in ALPHAS:
            model=fit(x[train],h[train],y[train],groups[train],k,seed+300+j,a)
            p=predict(model,x[valid],h[valid])
            errors[a].append(float(np.mean(((p-y[valid])/model['ys'])**2)))
    a=min(ALPHAS,key=lambda a:(np.mean(errors[a]),-a))
    return fit(x,h,y,groups,k,seed,a),a,{str(a):float(np.mean(v)) for a,v in errors.items()}

def external_inputs():
    from gse_caf_mechanism_runtime_20261017 import project_external
    genes_train=np.load(BASE/'data/gse176078_brca_genes.npy').astype(str)
    for folder in sorted((MECHANISM/'external/source').iterdir()):
        if not (folder/'manifest.json').exists():continue
        counts=sp.load_npz(folder/'counts.npz').tocsr();genes=np.load(folder/'genes.npy').astype(str)
        frame=pd.read_csv(folder/'metadata.csv',dtype=str);caf=np.flatnonzero(frame.label.to_numpy()=='Fibroblasts')
        lookup={g:j for j,g in enumerate(genes)}
        destination=np.array([j for j,g in enumerate(genes_train) if g in lookup])
        source=np.array([lookup[genes_train[j]] for j in destination])
        normalized=normalize(counts[caf]);coo=normalized[:,source].tocoo()
        aligned=sp.csr_matrix((coo.data,(coo.row,destination[coo.col])),shape=(len(caf),len(genes_train)))
        yield folder,counts,genes,frame,caf,aligned,project_external

def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    write(ROOT/'protocol.json',dict(arms=ARMS,source_seeds=SEEDS,folds=list(range(1,6)),
        primary_comparison='model - svd_prior (identical upstream expression features and biological program source)',
        secondary_comparisons=['model - pca_prior','model - pca'],primary_functions=['ECM','contractile','interferon'],
        pca='Centered logCP10k, all same 29733 genes, no gene variance scaling; 50 PCs standardized using outer training cells only',
        inner_scope='Only readout selection is inner donor-crossfit. Frozen upstream representations/PCA use outer training donors, not fully nested representation fitting.',
        primary_endpoint='Patient-within panel B Spearman correlation, paired differences; >=30 CAF; average seeds before patient inference',
        external_scope='Previously explored cohort, frozen-source transfer; not a pristine blind validation',
        marker_limitation='Panel B excluded from readout targets, not excluded from original expression/model inputs',
        sha256={p:sha(Path(__file__).parent/p) for p in [Path(__file__).name,'gse_caf_continuous_20261017.py','gse_caf_mechanism_runtime_20261017.py']}))
    counts=sp.load_npz(BASE/'data/gse176078_brca_counts.npz').tocsr()
    genes=np.load(BASE/'data/gse176078_brca_genes.npy').astype(str)
    normalized=normalize(counts);panels,_=read_markers(normalized,genes)
    for fold in range(1,6):
        note('fit_centered_PCA',fold=fold)
        with np.load(BASE/f'data/fold_{fold:02d}/biomed/gse176078_brca.npz') as d:
            train_idx=d['train_idx'].copy();features=d['features'].copy();patients=d['patient_ids'].astype(str)
            barcodes=d['barcodes'].astype(str)
        pca=centered_pca(normalized,train_idx,fold)
        with np.load(BASE/f'state/fold_{fold:02d}/biostate.npz') as d:prior=d['prior'].copy()
        for seed in SEEDS:
            case=ROOT/f'cases/fold_{fold:02d}/seed_{seed}';case.mkdir(parents=True,exist_ok=True)
            if (case/'complete.json').exists():continue
            with np.load(PREVIOUS/f'cases/fold_{fold:02d}/seed_{seed}/caf_representations.npz') as d:
                idx=d['idx'].copy();training=d['train'].copy();points=d['points'].copy();h=d['membership'].copy();k=float(d['curvature_k'])
            if set(patients[idx[training]])&set(patients[idx[~training]]):raise ValueError('Patient leakage')
            if not np.isin(idx[training],train_idx).all():raise ValueError('Training index mismatch')
            px=project(normalized[idx],pca);groups=patients[idx];models={}
            for arm in ARMS:
                note('readout_crossfit',fold=fold,seed=seed,arm=arm)
                x=points if arm=='model' else features[idx] if arm=='svd_prior' else px
                aux=h if arm=='model' else np.empty((len(idx),0)) if arm=='pca' else prior[idx]
                model,a,loss=select(x[training],aux[training],panels['A'][idx[training]],groups[training],k if arm=='model' else None,seed+fold)
                models[arm]=model
                np.savez_compressed(case/f'{arm}_readout.npz',**{key:value for key,value in model.items() if key!='k'},curvature_k=np.nan if model['k'] is None else model['k'])
                out=pd.DataFrame(dict(barcode=barcodes[idx[~training]],patient=groups[~training]))
                pred=predict(model,x[~training],aux[~training])
                for j,name in enumerate(CAF_SIGNATURES):
                    out['score_'+name]=pred[:,j];out['expression_'+name+'_B']=panels['B'][idx[~training],j]
                out.to_csv(case/f'{arm}_source.csv',index=False)
                write(case/f'{arm}_selection.json',dict(alpha=a,inner_losses=loss,checkpoint_sha256=sha(BASE/f'runs/state_hyper_cl/fold_{fold:02d}/seed_{seed}/checkpoint.pt')))
                # Exact replay ensures reused external model scores refer to the same readout.
                if arm=='model':
                    from gse_caf_mechanism_runtime_20261017 import readout_parameters,functional_scores
                    reference=functional_scores(readout_parameters(fold,seed),points[~training],h[~training])
                    if not np.allclose(pred,reference,atol=1e-7,rtol=1e-7):raise ValueError('Existing model readout replay differs')
            for folder,ec,eg,frame,caf,aligned,projector in external_inputs():
                note('external_frozen_projection',fold=fold,seed=seed,source=folder.name)
                svd,ep,_=projector(ec[caf],eg,fold);pp=project(aligned,pca)
                truth=pd.read_csv(MECHANISM/'external/CAF_frozen_ensemble_scores.csv',dtype={'barcode':str,'patient':str}).set_index('barcode')
                for arm in ARMS:
                    out=frame.iloc[caf][['barcode','patient']].reset_index(drop=True)
                    if arm=='model':
                        with np.load(MECHANISM/f'external/cases/{folder.name}/fold_{fold:02d}/seed_{seed}/CAF_predictions.npz') as d:
                            if not np.array_equal(out.barcode.to_numpy(str),d['barcodes'].astype(str)):raise ValueError('External alignment differs')
                            pred=d['scores'].copy()
                    else:
                        x=svd if arm=='svd_prior' else pp;aux=np.empty((len(caf),0)) if arm=='pca' else ep
                        pred=predict(models[arm],x,aux)
                    for j,name in enumerate(CAF_SIGNATURES):
                        out['score_'+name]=pred[:,j];out['expression_'+name+'_B']=truth.loc[out.barcode,'expression_'+name+'_B'].to_numpy()
                    out.to_csv(case/f'{arm}_external_{folder.name}.csv',index=False)
            write(case/'complete.json',dict(complete=True))
    summarize()

def partial_rho(x,y,q):
    x=rankdata(x);y=rankdata(y);q=np.column_stack([np.ones(len(q)),np.apply_along_axis(rankdata,0,q)])
    x=x-q@np.linalg.lstsq(q,x,rcond=None)[0];y=y-q@np.linalg.lstsq(q,y,rcond=None)[0]
    if np.std(x)<1e-8 or np.std(y)<1e-8:return None
    return float(np.corrcoef(x,y)[0,1])

def summarize():
    # Aggregate raw scores first, then use donors as the independent units.
    rows=[]
    meta=pd.read_csv(Path(__file__).parent.parent/'data/biomed/Wu_etal_2021_BRCA_scRNASeq/metadata.csv',index_col=0)
    qsource=pd.DataFrame({'barcode':meta.index.astype(str),'q1':np.log1p(meta.nCount_RNA.to_numpy()),'q2':np.log1p(meta.nFeature_RNA.to_numpy()),'q3':meta['percent.mito'].to_numpy()}).set_index('barcode')
    qexternal=[]
    for folder in sorted((MECHANISM/'external/source').iterdir()):
        if not (folder/'manifest.json').exists():continue
        c=sp.load_npz(folder/'counts.npz').tocsr();g=np.load(folder/'genes.npy').astype(str);m=pd.read_csv(folder/'metadata.csv',dtype=str)
        totals=np.asarray(c.sum(1)).ravel();mt=np.asarray(c[:,np.char.startswith(np.char.upper(g),'MT-')].sum(1)).ravel()
        qexternal.append(pd.DataFrame(dict(barcode=m.barcode,q1=np.log1p(totals),q2=np.log1p(c.getnnz(1)),q3=100*mt/np.maximum(totals,1))).set_index('barcode'))
    qexternal=pd.concat(qexternal)
    for cohort in ('source','external'):
        for arm in ARMS:
            files=list((ROOT/'cases').glob(f'fold_*/seed_*/{arm}_{cohort}*.csv'))
            data=pd.concat([pd.read_csv(p,dtype={'barcode':str,'patient':str}) for p in files],ignore_index=True)
            data=data.groupby(['barcode','patient'],as_index=False).mean(numeric_only=True)
            data.to_csv(ROOT/f'{cohort}_{arm}_ensemble.csv',index=False)
            qc=(qsource if cohort=='source' else qexternal).loc[data.barcode].to_numpy()
            for patient,ids in data.groupby('patient').groups.items():
                ids=np.asarray(list(ids));table=data.loc[ids]
                if len(table)<30:continue
                for name in CAF_SIGNATURES:
                    x=table['score_'+name].to_numpy();y=table['expression_'+name+'_B'].to_numpy();valid=np.isfinite(x)&np.isfinite(y)
                    if valid.sum()<30:continue
                    rows.append(dict(cohort=cohort,arm=arm,patient=patient,program=name,cells=int(valid.sum()),rho=rho(x[valid],y[valid]),qc_rho=partial_rho(x[valid],y[valid],qc[ids][valid])))
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'patient_correlations.csv',index=False)
    comparisons=[];rng=np.random.default_rng(176078018)
    for cohort in ('source','external'):
        for metric in ('rho','qc_rho'):
            for comparator in ('svd_prior','pca_prior','pca'):
                for name in CAF_SIGNATURES:
                    sub=frame[(frame.cohort==cohort)&(frame.program==name)].pivot(index='patient',columns='arm',values=metric)
                    sub=sub[['model',comparator]].dropna();delta=(sub.model-sub[comparator]).to_numpy()
                    if not len(delta):continue
                    interval=np.quantile(rng.choice(delta,(5000,len(delta)),replace=True).mean(1),[.025,.975])
                    p=float(wilcoxon(delta,method='approx').pvalue) if np.any(delta!=0) else 1.
                    comparisons.append(dict(cohort=cohort,metric=metric,comparator=comparator,program=name,patients=len(delta),mean_delta=float(delta.mean()),CI95=interval.tolist(),improved_fraction=float((delta>0).mean()),p=p))
    # Separate prespecified primary family: source/external x three functions, raw rho, exact-input comparator.
    for r in comparisons:r['primary']=r['comparator']=='svd_prior' and r['metric']=='rho' and r['program']!='inflammatory'
    for primary in (True,False):
        family=[r for r in comparisons if r['primary']==primary]
        for r,q in zip(family,bh([r['p'] for r in family])):r['BH_q']=float(q)
    write(ROOT/'paired_summary.json',comparisons)
    note('complete',cases=10,arms=list(ARMS),primary_tests=6)

def self_test():
    rng=np.random.default_rng(18);x=rng.normal(size=(90,8));g=np.repeat(['a','b','c'],30);y=x[:,:4]
    model=fit(x,np.empty((90,0)),y,g,None,4,1.)
    assert np.isfinite(predict(model,x,np.empty((90,0)))).all()
    from gse_caf_continuous_20261017 import fit_readout,predict as old_predict
    tangent=rng.normal(scale=.2,size=(90,5));p=np.column_stack([np.sqrt(3+(tangent*tangent).sum(1)),tangent]);h=rng.dirichlet(np.ones(5),90)
    a=fit(p,h,y,g,3.,4,1.);b=fit_readout(p,h,y,g,3.,1.,4)
    assert np.allclose(predict(a,p,h),old_predict(b,p,h),atol=1e-10)
    # Operator centering checked independently without writing experiment PCA artifacts.
    from scipy.linalg import svd
    centered=x-x.mean(0);_,s,_=svd(centered,full_matrices=False)
    op=LinearOperator(x.shape,matvec=lambda v:centered@v,rmatvec=lambda v:centered.T@v,dtype=float)
    _,ss,_=svds(op,k=4,random_state=18)
    assert np.allclose(np.sort(ss),np.sort(s[:4]))
    print('SELF_TEST_PASS: Lorentz readout exact replay, Euclidean readout, centered sparse SVD')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test()
    else:run()
