"""Frozen source classifiers applied per external sample, preserving all-cell context."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='4'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

MEDIUM=Path(__file__).resolve().parent
BASE=MEDIUM.parent/'bio_runs/gse176078_full_20261017'
BASE=BASE.resolve()
ROOT=BASE.parent/'gse176078_program_confirm_20261023'
EXTERNAL=BASE.parent/'gse176078_caf_mechanism_20261017/external/source'
FINAL=BASE.parent/'gse176078_caf_final_compare_20261018'
OLD=BASE.parent/'gse176078_caf_confirm_20261022'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read_metadata(path):
    # The LightHGNN environment intentionally has no pandas. Preserve CSV row
    # order and grouping using the standard library rather than changing it.
    with path.open(newline='',encoding='utf-8-sig') as handle:
        rows=list(csv.DictReader(handle))
    return {key:np.array([row[key] for row in rows],dtype=str)
            for key in ('barcode','label','sample')}


def run(method,seed):
    torch.set_num_threads(4)
    if method=='full':
        from gse_caf_mechanism_runtime_20261017 import load_full
        from gse_caf_final_compare_20261018 import final_embedding
        model=load_full(1,seed)
        checkpoint=BASE/f'runs/state_hyper_cl/fold_01/seed_{seed}/checkpoint.pt'
        device=torch.device('cpu')
    else:
        from gse_caf_confirm_worker_20261022 import BASELINES,factory,block
        device=torch.device('cuda:0' if method=='fhnn' and torch.cuda.is_available() else 'cpu')
        config=json.loads((OLD/f'selected/{method}/fold_01/seed_{seed}/selection.json').read_text())
        checkpoint=Path(config['checkpoint'])
        if method=='lighthgnn':
            sys.path.insert(0,str(BASELINES/'LightHGNN-main'))
            from my_model import MyMLPs
            model=MyMLPs(50,128,9).to(device)
            forward=lambda x,g:(model(x),model(x,get_emb=True))
        else:
            model,forward,*_=factory(method,device,seed,60766)
        model.load_state_dict(torch.load(checkpoint,map_location=device)['state_dict'],strict=True)
        model.eval()
        source_audit=BASE.parent/f'gse176078_program_classifiers_20261022/representations/{method}/seed_{seed}.json'
        assert json.loads(source_audit.read_text())['checkpoint_sha256']==sha(checkpoint)
    for folder in sorted(EXTERNAL.iterdir()):
        if not (folder/'manifest.json').exists():continue
        out=ROOT/f'external_representations/{method}/seed_{seed}/{folder.name}.npz'
        if out.exists():continue
        print(json.dumps(dict(stage='external_inference',method=method,seed=seed,object=folder.name)),flush=True)
        frame=read_metadata(folder/'metadata.csv')
        tumor=np.flatnonzero(frame['label']=='Malignant')
        positions=np.full(len(frame['label']),-1,np.int64);positions[tumor]=np.arange(len(tumor))
        cache=FINAL/f'external_inputs/{folder.name}/fold_01.npz'
        with np.load(cache) as z:x=z['features'].copy();prior=z['prior'].copy()
        points=None;audits=[]
        for sample in np.unique(frame['sample']):
            ids=np.flatnonzero(frame['sample']==sample);keep=positions[ids]>=0
            if not keep.any():continue
            edge_path=cache.parent/f'fold_01_{sample}_edges.npy';edges=np.load(edge_path)
            if method=='full':
                z,_,head_error=final_embedding(model,x[ids],prior[ids],np.zeros(len(ids),np.int64),edges)
            else:
                if method=='lighthgnn':
                    tx=torch.from_numpy(x[ids]).float().to(device);graph=None
                else:
                    local=dict(features=x[ids],labels=np.zeros(len(ids),np.int64),edge_index=edges)
                    tx,graph,_=block(local,np.arange(len(ids)),method,device)
                with torch.inference_mode():logits,embedding=forward(tx,graph)
                assert torch.isfinite(logits).all() and torch.isfinite(embedding).all()
                z=embedding.cpu().numpy();head_error=None
            if points is None:points=np.full((len(tumor),z.shape[1]),np.nan,np.float32)
            points[positions[ids[keep]]]=z[keep]
            audits.append(dict(sample=sample,context_cells=len(ids),tumor_cells=int(keep.sum()),
                classifier_head_error=head_error,graph_sha256=sha(edge_path)))
        assert points is not None and np.isfinite(points).all()
        out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,idx=tumor,points=points,barcodes=frame['barcode'][tumor])
        out.with_suffix('.json').write_text(json.dumps(dict(method=method,seed=seed,
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),device=str(device),
            input_sha256=sha(cache),metadata_sha256=sha(folder/'metadata.csv'),audits=audits,
            retraining=False,external_fitting=False),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--method',required=True)
    parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    run(args.method,args.seed)
