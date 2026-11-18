"""Continuous CAF program readouts from frozen Lorentz points and soft edges.

No state clustering, no classifier retraining. A four-output analytical readout
is fitted on training donors, selected by inner donor holdout, and evaluated on
outer held-out donors. Marker panels B are never readout fitting targets.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from gse_caf_analysis_20261017 import (
    BASE, MEDIUM, ROOT as PREVIOUS, SEEDS, CAF_SIGNATURES, signatures, dist, sha,
    write, bh, correlation_table,
)

ROOT=BASE.parent/'gse176078_caf_continuous_20261017'
PANELS={
    'ECM':dict(A=['COL1A1','COL1A2','COL3A1'],B=['FN1','POSTN','MMP11']),
    'inflammatory':dict(A=['CXCL12','C3','CFD'],B=['IL6','CCL2']),
    'contractile':dict(A=['ACTA2','TAGLN'],B=['MYL9','TPM2']),
    'interferon':dict(A=['ISG15','IFIT1','IFIT3'],B=['MX1','IFI6']),
}
ALPHAS=(.1,1.,10.,100.)


def note(stage,**fields):
    record=dict(stage=stage,time=datetime.now(timezone.utc).isoformat(),**fields)
    write(ROOT/'status.json',record)
    print(json.dumps(record),flush=True)


def equal_donor_weights(groups):
    _,code,n=np.unique(groups,return_inverse=True,return_counts=True)
    weights=1/n[code]
    return weights/weights.mean()


def fit_scaler(x,groups):
    w=equal_donor_weights(groups)
    mean=np.average(x,axis=0,weights=w)
    sd=np.sqrt(np.average((x-mean)**2,axis=0,weights=w))
    return mean,np.maximum(sd,1e-6)


def landmarks(points,groups,kappa,seed):
    rng=np.random.default_rng(seed)
    unique=np.unique(groups)
    chosen=[]
    # At most four cells per training donor, independent of marker values.
    for group in unique:
        ids=np.flatnonzero(groups==group)
        chosen.extend(rng.choice(ids,min(4,len(ids)),replace=False).tolist())
    selected=points[np.asarray(chosen)]
    d=dist(points,selected,kappa)
    bandwidth=max(float(np.median(d)),.05)
    return selected,bandwidth


def design(points,h,centers,bandwidth,kappa):
    d=dist(points,centers,kappa)
    # The only point-to-point metric is Lorentz geodesic distance with saved k.
    return np.column_stack([np.exp(-.5*(d/bandwidth)**2),h])


def fit_readout(points,h,y,groups,kappa,alpha,seed):
    centers,bandwidth=landmarks(points,groups,kappa,seed)
    x=design(points,h,centers,bandwidth,kappa)
    xm,xs=fit_scaler(x,groups)
    ym,ys=fit_scaler(y,groups)
    model=Ridge(alpha=alpha,solver='cholesky')
    model.fit((x-xm)/xs,(y-ym)/ys,sample_weight=equal_donor_weights(groups))
    return dict(centers=centers,bandwidth=bandwidth,kappa=kappa,xm=xm,xs=xs,ym=ym,ys=ys,model=model)


def predict(readout,points,h):
    x=design(points,h,readout['centers'],readout['bandwidth'],readout['kappa'])
    return readout['model'].predict((x-readout['xm'])/readout['xs'])*readout['ys']+readout['ym']


def inner_fit(points,h,y,groups,kappa,seed):
    predictions={a:np.full_like(y,np.nan,dtype=float) for a in ALPHAS}
    errors={a:[] for a in ALPHAS}
    for j,donor in enumerate(np.unique(groups)):
        validation=groups==donor
        training=~validation
        # Landmarks, preprocessing and target statistics are refit without the
        # inner-held-out donor. No outer test data enter inner model selection.
        for alpha in ALPHAS:
            model=fit_readout(points[training],h[training],y[training],groups[training],kappa,alpha,seed+300+j)
            p=predict(model,points[validation],h[validation])
            predictions[alpha][validation]=p
            errors[alpha].append(float(np.mean(((p-y[validation])/model['ys'])**2)))
    selected=min(ALPHAS,key=lambda a:(np.mean(errors[a]),-a))
    if not np.isfinite(predictions[selected]).all():raise ValueError('Inner donor predictions missing')
    return selected,predictions[selected],{str(a):float(np.mean(errors[a])) for a in ALPHAS}


def rho(x,y):
    if len(x)<5 or np.std(x)<1e-10 or np.std(y)<1e-10:return None
    value=float(spearmanr(x,y).statistic)
    return value if np.isfinite(value) else None


def centered_rho(x,y,groups):
    a=np.asarray(x,float).copy();b=np.asarray(y,float).copy()
    for group in np.unique(groups):
        mask=groups==group
        a[mask]-=a[mask].mean();b[mask]-=b[mask].mean()
    return rho(a,b)


def effective_rank(values):
    singular=np.linalg.svd(values-values.mean(0),compute_uv=False)
    energy=singular**2
    if energy.sum()<1e-15:return dict(numerical_rank=0,entropy_effective_rank=0.,energy_fraction=[])
    energy/=energy.sum()
    positive=energy[energy>1e-15]
    return dict(numerical_rank=int((singular>singular[0]*1e-6).sum()),
        entropy_effective_rank=float(np.exp(-(positive*np.log(positive)).sum())),energy_fraction=energy.tolist())


def read_markers(expression,genes):
    upper=np.char.upper(genes.astype(str))
    panels={};coverage={}
    for panel in ('A','B'):
        scores=[]
        for name in CAF_SIGNATURES:
            selected=np.flatnonzero(np.isin(upper,PANELS[name][panel]))
            if len(selected)<2:raise ValueError(f'Insufficient panel {panel} genes for {name}')
            scores.append(np.asarray(expression[:,selected].mean(1)).ravel())
            coverage[name+'_'+panel]=upper[selected].tolist()
        panels[panel]=np.column_stack(scores)
    return panels,coverage


def analyze_case(fold,seed,atlas,meta,panels):
    target=ROOT/f'cases/fold_{fold:02d}/seed_{seed}'
    target.mkdir(parents=True,exist_ok=True)
    previous=PREVIOUS/f'cases/fold_{fold:02d}/seed_{seed}'
    note('inner_donor_continuous_readout',fold=fold,seed=seed)
    with np.load(previous/'caf_representations.npz') as d:
        idx=d['idx'].copy();p=d['points'].copy();h=d['membership'].copy()
        training=d['train'].copy();kappa=float(d['curvature_k'])
        names=d['program_names'].astype(str)
    provenance=json.loads((previous/'provenance.json').read_text())
    case=BASE/f'runs/state_hyper_cl/fold_{fold:02d}/seed_{seed}'
    if sha(case/'checkpoint.pt')!=provenance['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    if abs(kappa-provenance['k'])>1e-8:raise ValueError('Mismatched Lorentz curvature')
    if not np.allclose(-p[:,0]**2+(p[:,1:]**2).sum(1),-kappa,atol=1e-7):raise ValueError('Invalid Lorentz points')
    with np.load(BASE/f'data/fold_{fold:02d}/biomed/gse176078_brca.npz') as d:
        caf=atlas['class_names'][atlas['labels']]=='CAFs'
        assert np.array_equal(np.sort(idx[training]),np.sort(d['train_idx'][caf[d['train_idx']]]))
        assert np.array_equal(np.sort(idx[~training]),np.sort(d['test_idx'][caf[d['test_idx']]]))
        assert np.array_equal(d['barcodes'],atlas['barcodes'])
    groups=atlas['patient_ids'][idx]
    if set(groups[training])&set(groups[~training]):raise ValueError('Outer patient leakage')
    targets=panels['A'][idx]
    alpha,crossfit,losses=inner_fit(p[training],h[training],targets[training],groups[training],kappa,seed+fold)
    readout=fit_readout(p[training],h[training],targets[training],groups[training],kappa,alpha,seed+fold)
    outer=predict(readout,p[~training],h[~training])
    # All scores retain gene-program identity and four separate output weights.
    ym,ys=fit_scaler(targets[training],groups[training])
    outer_z=(outer-ym)/ys
    crossfit_z=(crossfit-ym)/ys
    thresholds=np.quantile(crossfit_z,2/3,axis=0)
    quality=[]
    for j,name in enumerate(CAF_SIGNATURES):
        r=centered_rho(crossfit[:,j],targets[training,j],groups[training])
        variation=float(crossfit[:,j].std()/max(targets[training,j].std(),1e-6))
        quality.append(dict(program=name,inner_donor_centered_rho=r,relative_score_sd=variation,
            reliable_for_joint_state=bool(r is not None and r>=.2 and variation>=.1)))
    frame=pd.DataFrame(dict(global_idx=idx[~training],barcode=atlas['barcodes'][idx[~training]],
        patient=groups[~training],fold=fold,seed=seed))
    for j,name in enumerate(CAF_SIGNATURES):
        frame['latent_'+name]=outer_z[:,j]
        frame['readout_logCP10k_'+name]=outer[:,j]
        frame['panel_A_'+name]=panels['A'][idx[~training],j]
        frame['panel_B_'+name]=panels['B'][idx[~training],j]
    # Failure of a program readout is missing biological evidence, not zero
    # joint-high cells. The rule uses inner donors only, never outer markers.
    eligible=quality[0]['reliable_for_joint_state'] and quality[1]['reliable_for_joint_state']
    joint=((outer_z[:,0]>=thresholds[0])&(outer_z[:,1]>=thresholds[1]))
    frame['joint_ECM_inflammatory_available']=eligible
    frame['joint_ECM_inflammatory_high']=pd.array(joint,dtype='boolean')
    if not eligible:frame['joint_ECM_inflammatory_high']=pd.NA
    for column in ('nCount_RNA','nFeature_RNA','percent.mito'):
        frame[column]=meta.iloc[idx[~training]][column].to_numpy()
    frame.to_csv(target/'caf_test_continuous.csv',index=False)
    np.savez_compressed(target/'continuous_readout.npz',landmarks_lorentz=readout['centers'],
        curvature_k=kappa,bandwidth=readout['bandwidth'],feature_mean=readout['xm'],feature_sd=readout['xs'],
        target_mean=readout['ym'],target_sd=readout['ys'],coefficients=readout['model'].coef_,
        intercept=readout['model'].intercept_,program_names=np.asarray(CAF_SIGNATURES),
        hallmark_names=names,joint_high_thresholds=thresholds,alpha=alpha)
    rows=[]
    for group in np.unique(groups[~training]):
        mask=groups[~training]==group
        for j,name in enumerate(CAF_SIGNATURES):
            for panel in ('A','B'):
                value=rho(outer[mask,j],panels[panel][idx[~training][mask],j])
                rows.append(dict(patient=group,fold=fold,seed=seed,program=name,panel=panel,cells=int(mask.sum()),rho=value))
    pd.DataFrame(rows).to_csv(target/'patient_program_recovery.csv',index=False)
    diagnostics=[]
    for j,name in enumerate(CAF_SIGNATURES):
        for column in ('nCount_RNA','nFeature_RNA','percent.mito'):
            diagnostics.append(dict(program=name,variable=column,
                centered_rho=centered_rho(outer_z[:,j],frame[column].to_numpy(),frame.patient.to_numpy())))
    write(target/'diagnostics.json',dict(stage='complete',chosen_alpha=alpha,inner_patient_standardized_MSE=losses,
        quality=quality,prediction_rank=effective_rank(outer_z),inner_prediction_rank=effective_rank(crossfit_z),
        program_correlations=np.corrcoef(outer_z.T).tolist(),QC=diagnostics,
        joint_available=bool(eligible),joint_high_count=int(joint.sum()) if eligible else None,
        selection='Inner leave-one-training-donor-out standardized MSE; landmarks refitted in each inner split',
        panel_B_used_for_fit=False,outer_test_used_for_selection=False,
        panel_B_independent_of_original_model_inputs=False,
        geometry='Lorentz RBF distances from saved k; no Poincare conversion and no Euclidean cosine on manifold points',
        saved_representation_sha256=sha(previous/'caf_representations.npz'),checkpoint_sha256=provenance['checkpoint_sha256']))
    return frame,rows,h[~training],names


def summarize_recovery(recovery,seed):
    rows=[];rng=np.random.default_rng(seed)
    for (name,panel),d in recovery.groupby(['program','panel']):
        available=d[d.cells>=30].dropna(subset=['rho'])
        values=available.rho.to_numpy()
        boot=np.mean(rng.choice(values,(2000,len(values)),replace=True),axis=1) if len(values) else None
        rows.append(dict(program=name,panel=panel,patients=len(available),
            mean_within_patient_rho=float(values.mean()) if len(values) else None,
            median_within_patient_rho=float(np.median(values)) if len(values) else None,
            positive_patient_fraction=float((values>0).mean()) if len(values) else None,
            patient_bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist() if boot is not None else None,
            role='Same fitting marker panel A' if panel=='A' else 'Disjoint descriptive marker panel B; original model saw these genes',
            uncertainty='Conditional on frozen readouts; patients, not cells, resampled'))
    return rows


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    note('input_audit')
    if json.loads((PREVIOUS/'status.json').read_text())['stage']!='complete_exploratory_stage':raise ValueError('Previous exports incomplete')
    with np.load(BASE/'data/gse176078_brca.npz') as d:atlas={k:d[k].copy() for k in d.files}
    meta=pd.read_csv(BASE.parent.parent/'data/biomed/Wu_etal_2021_BRCA_scRNASeq/metadata.csv',low_memory=False)
    assert np.array_equal(meta.iloc[:,0].to_numpy(str),atlas['barcodes'])
    genes=np.load(BASE/'data/gse176078_brca_genes.npy').astype(str)
    counts=sp.load_npz(BASE/'data/gse176078_brca_counts.npz').tocsr()
    expression=counts.multiply((1e4/np.maximum(np.asarray(counts.sum(1)).ravel(),1))[:,None]).tocsr()
    expression.data=np.log1p(expression.data)
    panels,coverage=read_markers(expression,genes)
    immune,_=signatures(expression,genes)
    plan=dict(dataset='GSE176078 all 100064 cells; all 6573 CAF',seeds=list(SEEDS),folds=list(range(1,6)),
        model='Existing full biological state_hyper_cl checkpoints; no changes or classifier retraining',
        representation='Continuous saved 33D Lorentz cell coordinates plus 50 learned soft memberships',
        readout='Four-output donor-weighted ridge on Lorentz landmark RBF and native soft memberships',
        training='Outer training donors only; inner leave-one-donor-out refits landmarks, scalers and targets for alpha selection',
        alpha_grid=ALPHAS,markers=PANELS,coverage=coverage,
        no_clustering=True,no_test_association_tuning=True,
        marker_independence='B disjoint from A readout targets; both may have entered original full-model features and Hallmark priors',
        primary_functional_assessment='Per-patient held-out recovery of four panel-B scores; all four reported',
        joint_rule='Both cross-fitted training upper-tertile thresholds exceeded; only if both inner centered rho>=.2 and relative sd>=.1; otherwise NA',
        patient_associations='Same three original planned associations, cohort >=30 CAF and >=30 relevant immune cells; exploratory repeated analysis',
        uncertainty='Patient bootstrap conditional on fitted readouts; no cell independence or doubling patients across seeds',
        limitations=['Readout target calibration is marker-guided; cannot call this unsupervised novel-state discovery.',
            'Original model saw both marker panels. This does not establish held-out-gene generalization.',
            'Soft memberships sum to one and candidates are top-three prior constrained. Their associations can reflect compositional constraints.',
            'No new model comparison, no causal or clinical claims, no external-cohort or spatial confirmation yet.',
            'Program names reflect prior sets; native edge recurrence alone is not novel biological evidence.'],
        previous_result_directory=str(PREVIOUS),source_sha256=sha(__file__),helper_sha256=sha(MEDIUM/'gse_caf_analysis_20261017.py'),
        full_model_plan_sha256=sha(BASE/'runs/plan.json'))
    write(ROOT/'plan.json',plan)
    shutil.copy2(__file__,ROOT/Path(__file__).name)
    all_outputs={};combined=[]
    for seed in SEEDS:
        frames=[];recoveries=[];hs=[]
        for fold in range(1,6):
            frame,records,h,names=analyze_case(fold,seed,atlas,meta,panels)
            frames.append(frame);recoveries.extend(records);hs.append(h)
        oof=pd.concat(frames,ignore_index=True)
        memberships=np.concatenate(hs)
        order=np.argsort(oof.global_idx.to_numpy())
        oof=oof.iloc[order].reset_index(drop=True);memberships=memberships[order]
        expected=np.flatnonzero(atlas['class_names'][atlas['labels']]=='CAFs')
        assert np.array_equal(oof.global_idx.to_numpy(),expected)
        folder=ROOT/f'summary/seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
        oof.to_csv(folder/'caf_out_of_fold_continuous.csv',index=False)
        np.savez_compressed(folder/'caf_out_of_fold_soft_edges.npz',global_idx=expected,program_names=names,memberships=memberships)
        recovery=pd.DataFrame(recoveries);recovery.to_csv(folder/'patient_program_recovery.csv',index=False)
        recovery_summary=summarize_recovery(recovery,seed)
        write(folder/'functional_recovery_summary.json',recovery_summary)
        patients=[];native=[]
        for patient,table in oof.groupby('patient'):
            selected=atlas['patient_ids']==patient
            mac=np.flatnonzero(selected&(atlas['celltype_minor']=='Macrophage'))
            cd8=np.flatnonzero(selected&(atlas['celltype_minor']=='T cells CD8+'))
            subtype=np.unique(atlas['subtypes'][selected]);assert len(subtype)==1
            eligible=bool(table.joint_ECM_inflammatory_available.all())
            row=dict(patient=patient,subtype=subtype[0],caf_cells=len(table),macrophage_cells=len(mac),CD8_cells=len(cd8),
                caf_mean_log_umi=float(np.log1p(table.nCount_RNA).mean()),caf_mean_mito=float(table['percent.mito'].mean()),
                joint_high_available=eligible,joint_high_fraction=float(table.joint_ECM_inflammatory_high.mean()) if eligible else np.nan)
            # Common marker-expression units across folds; do not compare
            # donor means in separately standardized fold coordinate systems.
            for name in CAF_SIGNATURES:row['latent_'+name]=float(table['readout_logCP10k_'+name].mean())
            for name in ('macrophage_TREM2','macrophage_lipid','CD8_cytotoxic','CD8_exhaustion_related'):
                ids=mac if name.startswith('macrophage') else cd8
                row[name]=float(immune[name][ids].mean()) if len(ids) else np.nan
            patients.append(row)
            mask=oof.patient.to_numpy()==patient
            for j,name in enumerate(names):
                weights=memberships[mask,j]
                native.append(dict(patient=patient,program=name,caf_cells=int(mask.sum()),
                    support=int((weights>0).sum()),mean_soft_weight=float(weights.mean()),
                    note='Native simplex weight; descriptive, not absolute program activity'))
        patient_frame=pd.DataFrame(patients);patient_frame.to_csv(folder/'patient_summary.csv',index=False)
        pd.DataFrame(native).to_csv(folder/'native_hyperedge_weights.csv',index=False)
        note('patient_associations',seed=seed)
        associations=correlation_table(patient_frame,seed)
        write(folder/'associations.json',associations)
        combined.extend(associations);all_outputs[seed]=oof
    a,b=all_outputs.values();stability=[]
    for name in CAF_SIGNATURES:
        stability.append(dict(program=name,cell_score_spearman=rho(a['latent_'+name],b['latent_'+name])))
    valid_joint=a.joint_ECM_inflammatory_available & b.joint_ECM_inflammatory_available
    stability.append(dict(program='joint_ECM_inflammatory',available_in_both_cells=int(valid_joint.sum()),
        cells_high_seed0=int(a.loc[valid_joint,'joint_ECM_inflammatory_high'].sum()) if valid_joint.any() else None,
        cells_high_seed1=int(b.loc[valid_joint,'joint_ECM_inflammatory_high'].sum()) if valid_joint.any() else None,
        agreement=float((a.loc[valid_joint,'joint_ECM_inflammatory_high']==b.loc[valid_joint,'joint_ECM_inflammatory_high']).mean()) if valid_joint.any() else None))
    write(ROOT/'summary/seed_sensitivity.json',stability)
    write(ROOT/'summary/associations_all_seeds.json',combined)
    note('complete_continuous_exploration',cases=10,CAF_cells=6573,
        next='Review held-out marker recovery, dimensionality, technical confounding and seed stability before any external/spatial validation')


def self_test():
    rng=np.random.default_rng(23);kappa=3.
    x=rng.normal(size=(240,4))*.4
    points=np.column_stack([np.sqrt(kappa+(x*x).sum(1)),x])
    h=np.column_stack([1/(1+np.exp(-x)),1/(1+np.exp(x))])
    h/=h.sum(1,keepdims=True)
    groups=np.repeat(np.arange(8),30)
    selected=groups<6
    model=fit_readout(points[selected],h[selected],x[selected],groups[selected],kappa,.1,17)
    predicted=predict(model,points[~selected],h[~selected])
    assert effective_rank(predicted)['numerical_rank']==4
    assert min(rho(predicted[:,j],x[~selected,j]) for j in range(4))>.8
    assert np.allclose(-model['centers'][:,0]**2+(model['centers'][:,1:]**2).sum(1),-kappa)
    # Equal donor contribution even with unequal cell counts.
    uneven=np.r_[np.zeros(10),np.ones(50)]
    w=equal_donor_weights(uneven)
    assert np.isclose(w[:10].sum(),w[10:].sum())
    print(json.dumps(dict(four_independent_continuous_axes=True,Lorentz_geometry=True,donor_weighting=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        try:run()
        except Exception as error:note('failed',error=repr(error));raise
