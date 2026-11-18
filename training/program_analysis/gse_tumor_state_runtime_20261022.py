"""Frozen full-cohort inference for malignant-state discovery, not retraining."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from gse_caf_analysis_20261017 import BASE,write,sha
from gse_caf_mechanism_runtime_20261017 import load_full
from gse_caf_final_compare_20261018 import final_embedding,ROOT as FINAL

ROOT=BASE.parent/'gse176078_tumor_states_20261022'
FOLD=1
SEEDS=(168023000,168023001)


def event(stage,**kw):
    from datetime import datetime,timezone
    item=dict(stage=stage,time=datetime.now(timezone.utc).isoformat(),**kw)
    write(ROOT/'status.json',item);print(json.dumps(item),flush=True)


def source_data():
    with np.load(BASE/'data/fold_01/biomed/gse176078_brca.npz') as z:return {k:z[k].copy() for k in z.files}


def source_export(seed):
    output=ROOT/f'representations/seed_{seed}/source.npz'
    if output.exists():return output
    event('export_source_tumor',seed=seed)
    data=source_data()
    with np.load(BASE/'state/fold_01/biostate.npz') as z:prior=z['prior'].copy()
    tumor=data['class_names'][data['labels']].astype(str)=='Cancer Epithelial'
    model=load_full(1,seed);torch.set_num_threads(4)
    records=[];audit=[]
    with np.load(BASE/f'runs/state_hyper_cl/fold_01/seed_{seed}/predictions.npz') as z:
        reference=np.full((len(tumor),9),np.nan);reference[z['test_idx']]=z['prob']
    for role,key in enumerate(('train_idx','valid_idx','test_idx')):
        indices=data[key]
        blocks=[indices] if role==0 else [indices[data['patient_ids'][indices]==p] for p in np.unique(data['patient_ids'][indices])]
        for ids in blocks:
            if not tumor[ids].any():continue
            lookup=np.full(len(tumor),-1,dtype=np.int64);lookup[ids]=np.arange(len(ids))
            edges=data['edge_index'];keep=(lookup[edges[0]]>=0)&(lookup[edges[1]]>=0)
            _,groups=np.unique(data['patient_ids'][ids],return_inverse=True)
            z,prob,error=final_embedding(model,data['features'][ids],prior[ids],groups,lookup[edges[:,keep]])
            drift=float(np.abs(prob-reference[ids]).max()) if role==2 else None
            if drift is not None and drift>2e-4:raise ValueError(('Frozen prediction drift',drift))
            mask=tumor[ids]
            records.append((ids[mask],z[mask].astype(np.float32),np.full(mask.sum(),role,dtype=np.int8)))
            audit.append(dict(role=role,context_cells=len(ids),tumor_cells=int(mask.sum()),classifier_head_error=error,saved_test_probability_error=drift))
    idx=np.concatenate([r[0] for r in records]);assert len(np.unique(idx))==int(tumor.sum())==24489
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,idx=idx,points=np.concatenate([r[1] for r in records]),role=np.concatenate([r[2] for r in records]))
    write(output.with_suffix('.json'),dict(audits=audit,representation='Frozen Euclidean final classifier-input vector, not raw Lorentz coordinates',
        source_checkpoint_sha256=sha(BASE/f'runs/state_hyper_cl/fold_01/seed_{seed}/checkpoint.pt')))
    return output


def external_export(seed):
    from gse_caf_pca_compare_20261018 import MECHANISM
    model=load_full(1,seed);torch.set_num_threads(4)
    for folder in sorted((MECHANISM/'external/source').iterdir()):
        if not (folder/'manifest.json').exists():continue
        output=ROOT/f'representations/seed_{seed}/external_{folder.name}.npz'
        if output.exists():continue
        event('export_external_tumor',seed=seed,folder=folder.name)
        frame=pd.read_csv(folder/'metadata.csv',dtype=str)
        tumor=np.flatnonzero(frame.label.to_numpy()=='Malignant')
        with np.load(FINAL/f'external_inputs/{folder.name}/fold_01.npz') as z:
            x,prior=z['features'].copy(),z['prior'].copy()
        points=np.full((len(tumor),512),np.nan,dtype=np.float32)
        positions=np.full(len(frame),-1,dtype=np.int64);positions[tumor]=np.arange(len(tumor))
        for sample,part in frame.groupby('sample'):
            ids=part.index.to_numpy();mask=positions[ids]>=0
            if not mask.any():continue
            edges=np.load(FINAL/f'external_inputs/{folder.name}/fold_01_{sample}_edges.npy')
            z,_,_=final_embedding(model,x[ids],prior[ids],np.zeros(len(ids),dtype=np.int64),edges)
            points[positions[ids[mask]]]=z[mask]
        assert np.isfinite(points).all()
        np.savez_compressed(output,idx=tumor,points=points,barcodes=frame.iloc[tumor].barcode.to_numpy(str),
            patients=frame.iloc[tumor].patient.to_numpy(str),samples=frame.iloc[tumor]['sample'].to_numpy(str),
            subtype=np.full(len(tumor),{'SeuratObject_ERTotal':'ER+','SeuratObject_HER2':'HER2+','SeuratObject_TNBC':'TNBC'}[folder.name]))
