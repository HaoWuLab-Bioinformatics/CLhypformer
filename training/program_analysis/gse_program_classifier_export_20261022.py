"""Export already trained classifier representations; no downstream retraining."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='4'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from gse_caf_confirm_worker_20261022 import BASE, BASELINES, factory, block

MEDIUM=Path(__file__).resolve().parent
ROOT=BASE.parent/'gse176078_program_classifiers_20261022'
OLD=BASE.parent/'gse176078_caf_confirm_20261022'
PROGRAM=BASE.parent/'gse176078_patient_programs_20261022'


def run(method,seed):
    out=ROOT/f'representations/{method}/seed_{seed}.npz'
    if out.exists():return
    out.parent.mkdir(parents=True,exist_ok=True)
    selected=OLD/f'selected/{method}/fold_01/seed_{seed}'
    config=json.loads((selected/'selection.json').read_text())
    checkpoint=Path(config['checkpoint'])
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    torch.set_num_threads(4)
    # Use the training export's backend for the stored-logit check.
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with np.load(BASE/'data/fold_01/biomed/gse176078_brca.npz') as z:
        data={k:z[k].copy() for k in z.files}
    if method=='lighthgnn':
        sys.path.insert(0,str(BASELINES/'LightHGNN-main'))
        from my_model import MyMLPs
        model=MyMLPs(50,128,9).to(device)
        def forward(x, graph):
            return model(x), model(x,get_emb=True)
    else:
        model,forward,*_=factory(method,device,seed,len(data['train_idx']))
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict'],strict=True)
    model.eval()
    records,audits=[],[]
    for role,key in enumerate(('train_idx','valid_idx','test_idx')):
        ids=data[key]
        if method=='lighthgnn':
            x=torch.from_numpy(data['features'][ids]).float().to(device)
            graph=None
        else:
            x,graph,_=block(data,ids,method,device)
        with torch.inference_mode():logits,z=forward(x,graph)
        logits=logits.cpu().numpy();z=z.cpu().numpy()
        assert np.isfinite(logits).all() and np.isfinite(z).all()
        error=None
        if role:
            with np.load(selected/('validation.npz' if role==1 else 'test.npz')) as saved:
                assert np.array_equal(saved['idx'],ids)
                error=float(np.abs(logits-saved['scores']).max())
                assert error<2e-4,('Classifier replay mismatch',method,error)
        keep=data['class_names'][data['labels'][ids]].astype(str)=='Cancer Epithelial'
        records.append((ids[keep],z[keep].astype(np.float32),np.full(keep.sum(),role,np.int8)))
        audits.append(dict(role=role,context_cells=len(ids),replay_max_absolute_error=error))
    np.savez_compressed(out,idx=np.concatenate([r[0] for r in records]),
        points=np.concatenate([r[1] for r in records]),role=np.concatenate([r[2] for r in records]))
    out.with_suffix('.json').write_text(json.dumps(dict(method=method,seed=seed,
        checkpoint=str(checkpoint),checkpoint_sha256=digest,retraining=False,audits=audits,device=str(device),
        representation='Classifier hidden vector; FHNN uses spatial Lorentz logmap0 tangent coordinates'),indent=2))
    print(json.dumps(dict(stage='export_complete',method=method,seed=seed,audits=audits)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--method',required=True)
    parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    run(args.method,args.seed)
