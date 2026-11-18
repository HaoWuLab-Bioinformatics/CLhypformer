"""Inference-only message interventions; existing training files stay frozen."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
from gse_biostate_model import BiologicalStateHypergraph
from gse_lorentz_runtime import lorentz_cdist
from gse_shared_eval import original_args, sha256
from gse_caf_continuous_20261017 import BASE, ROOT as CONTINUOUS, design


class MessageIntervention(BiologicalStateHypergraph):
    def forward(self,x,prior,patients,global_idx=None,message_scale=None):
        z=self.bound(self.input(x))
        anchor=self.bound(self.input(self.expression_templates)+self.anchors)
        distance=lorentz_cdist(self.point(z),self.point(anchor),self.manifold.k).to(x.dtype)
        logits=prior.to(x.dtype).clamp_min(1e-8).log()-distance/.5
        logits=logits.masked_fill(prior<=0,-1e4)
        h=logits.softmax(1)*(prior>0)
        h=h/h.sum(1,keepdim=True).clamp_min(1e-8)
        scale=torch.ones(self.programs,device=x.device,dtype=x.dtype) if message_scale is None else message_scale
        if scale.shape!=(self.programs,) or (scale<0).any() or (scale>1).any():raise ValueError('Invalid message gate')
        updated=torch.empty_like(z);state_message=torch.zeros_like(z);edges=[];supports=[]
        for patient in torch.unique(patients,sorted=True):
            ids=torch.where(patients==patient)[0];local=z[ids];membership=h[ids]
            mass=membership.sum(0);edge=anchor
            for _ in range(2):
                aggregate=(membership.T@local)/mass[:,None].clamp_min(1e-6)
                edge=self.bound(edge+.25*torch.tanh(self.edge_update(torch.cat((aggregate,edge,anchor),dim=-1))))
                # Membership, normalizers, priors and graph are untouched.
                message=membership@(edge*scale[:,None])
                local=self.bound(local+.25*torch.tanh(self.node_update(torch.cat((local,message),dim=-1))))
            updated[ids]=local;state_message[ids]=membership@(edge*scale[:,None])
            edges.append(edge);supports.append((membership>0).sum(0))
        edge_tangent=torch.stack(edges)
        fused=x+torch.sigmoid(self.gate)*self.output_norm(self.output(updated))
        semantic=self.state_norm(self.state_output(state_message+updated-z))
        semantic=semantic*(h.sum(1,keepdim=True)>0)
        fused=fused+torch.sigmoid(self.state_gate)*semantic
        self.last=dict(cells=self.point(updated+state_message),edges=self.point(edge_tangent),
            edge_tangent=edge_tangent,membership=h,prior=prior,support=torch.stack(supports),fused=fused,original=z)
        return fused


def load_full(fold,seed):
    from hypformer import HypFormer
    plan=json.loads((BASE/'runs/plan.json').read_text())
    for name in ('hypformer.py','gse_biostate_model.py'):
        if sha256(Path(__file__).parent/name)!=plan['code_sha256'][name]:raise ValueError('Frozen source differs: '+name)
    case=BASE/f'runs/state_hyper_cl/fold_{fold:02d}/seed_{seed}'
    checkpoint=torch.load(case/'checkpoint.pt',map_location='cpu',weights_only=False)['state_dict']
    args=original_args(50,9)
    for name,value in dict(device=torch.device('cpu'),use_cl=1,use_adaptive_curvature=False,
        cl_mode='patient_prototype',hyperedge_policy='patient_state',prototype_temperature=.2,
        cl_weight=.001,cl_fusion_weight=0.,cl_aux_head=0,cl_consistency=0,
        cl_consistency_weight=.2,cl_fine_labels=0,cl_cross_patient=1,
        cl_pair_policy='diag_fixed',cl_mask_same_major=0,cl_patient_balance=0,
        cl_knn=5,cl_drop_rate=.2,cl_tau=.7,knn_metric='hyp',drop_strategy='random',
        cl_space='hyp',graph_build_method='knn',graph_source='learned',cl_feature_noise=.05,save_topology=0).items():
        setattr(args,name,value)
    model=HypFormer(args)
    model.state_module=MessageIntervention(50,len(checkpoint['state_module.anchors']),model.manifold_hidden,mode='nonlinear')
    for key in ('state_positive','state_negative'):model.register_buffer(key,checkpoint[key].clone())
    model.load_state_dict(checkpoint,strict=True);model.eval()
    return model


def load_state(fold,seed):
    from manifolds.lorentz import Lorentz
    path=BASE/f'runs/state_hyper_cl/fold_{fold:02d}/seed_{seed}/checkpoint.pt'
    ck=torch.load(path,map_location='cpu',weights_only=False)['state_dict']
    weights={k[len('state_module.'):]:v for k,v in ck.items() if k.startswith('state_module.')}
    module=MessageIntervention(50,len(weights['anchors']),Lorentz(k=float(weights['manifold.k'])),mode='nonlinear')
    module.load_state_dict(weights,strict=True);module.eval();return module


def readout_parameters(fold,seed):
    with np.load(CONTINUOUS/f'cases/fold_{fold:02d}/seed_{seed}/continuous_readout.npz') as d:
        return {k:d[k].copy() for k in d.files}


def functional_scores(parameters,points,memberships):
    x=design(points,memberships,parameters['landmarks_lorentz'],float(parameters['bandwidth']),float(parameters['curvature_k']))
    return (((x-parameters['feature_mean'])/parameters['feature_sd'])@parameters['coefficients'].T+parameters['intercept'])*parameters['target_sd']+parameters['target_mean']


def project_external(counts,genes,fold):
    """Frozen source SVD/scaler; external library totals use all measured genes."""
    from scipy import sparse
    training_genes=np.load(BASE/'data/gse176078_brca_genes.npy').astype(str)
    lookup={gene:j for j,gene in enumerate(genes)}
    destination=np.array([j for j,gene in enumerate(training_genes) if gene in lookup],dtype=int)
    source=np.array([lookup[training_genes[j]] for j in destination],dtype=int)
    normalized=counts.multiply((1e4/np.maximum(np.asarray(counts.sum(1)).ravel(),1))[:,None]).tocsr()
    normalized.data=np.log1p(normalized.data)
    with np.load(BASE/f'data/fold_{fold:02d}/biomed/preprocessing.npz') as d:
        components=d['components'];mean=d['mean'];scale=d['scale']
    features=(normalized[:,source]@components[:,destination].T-mean)/scale
    features=np.asarray(features,dtype=np.float32)
    with np.load(BASE/f'state/fold_{fold:02d}/biostate.npz') as d:
        state={k:d[k].copy() for k in ('selected_gene_names','gene_training_mean','gene_training_sd','program_training_mean','program_training_sd','program_gene_counts','program_names')}
    selected=state['selected_gene_names'].astype(str)
    expression=np.zeros((counts.shape[0],len(selected)),dtype=np.float32)
    observed=np.array([j for j,g in enumerate(selected) if g in lookup],dtype=int)
    cols=np.array([lookup[selected[j]] for j in observed],dtype=int)
    expression[:,observed]=normalized[:,cols].toarray()
    gene_z=np.clip((expression-state['gene_training_mean'])/state['gene_training_sd'],-5,5)
    gmt=BASE/'state/MSigDB_Hallmark_2020.gmt';programs={}
    for line in gmt.read_text().splitlines():
        fields=line.split('\t');programs[fields[0]]={g.split(',')[0].upper() for g in fields[2:]}
    masks=np.stack([np.isin(np.char.upper(selected),list(programs[n])) for n in state['program_names'].astype(str)])
    weights=masks.astype(np.float32)/state['program_gene_counts'][:,None]
    score=(gene_z@weights.T-state['program_training_mean'])/state['program_training_sd']
    prior=np.maximum(score-.5,0).astype(np.float32)
    chosen=np.argsort(prior,axis=1)[:,-3:];kept=np.zeros_like(prior)
    np.put_along_axis(kept,chosen,np.take_along_axis(prior,chosen,axis=1),axis=1)
    kept/=np.maximum(kept.sum(1,keepdims=True),1e-8)
    panel_coverage=(masks[:,observed].sum(1)/np.maximum(masks.sum(1),1)).tolist()
    coverage=dict(measured_training_gene_fraction=len(destination)/len(training_genes),
        source_gene_count=len(genes),missing_gene_policy='Zero expression for unavailable genes, explicitly recorded; no external refit',
        loading_energy_fraction=(np.square(components[:,destination]).sum(1)/np.square(components).sum(1)).tolist(),
        hallmark_measured_fraction=panel_coverage,program_names=state['program_names'].tolist())
    return features,kept,coverage


def self_test():
    from manifolds.lorentz import Lorentz
    torch.manual_seed(17);torch.set_num_threads(2)
    normal=BiologicalStateHypergraph(6,5,Lorentz(k=3),mode='nonlinear')
    changed=MessageIntervention(6,5,Lorentz(k=3),mode='nonlinear')
    changed.load_state_dict(normal.state_dict())
    x=torch.randn(60,6);prior=torch.rand(60,5);prior[prior<.5]=0
    groups=torch.zeros(60,dtype=torch.long)
    with torch.no_grad():
        a=normal(x,prior,groups);b=changed(x,prior,groups)
        assert torch.allclose(a,b,atol=1e-7)
        baseline_h=changed.last['membership'].clone()
        gate=torch.ones(5);gate[0]=.8
        c=changed(x,prior,groups,message_scale=gate)
        assert torch.equal(baseline_h,changed.last['membership'])
        assert (b-c).abs().max()>0
        points=changed.last['cells']
        assert torch.allclose(-points[:,0]**2+(points[:,1:]**2).sum(1),torch.full((60,),-3.,dtype=torch.float64),atol=1e-7)
    print(json.dumps(dict(zero_intervention_matches_original=True,membership_unchanged=True,Lorentz_constraints=True)))


if __name__=='__main__':
    self_test()
    import sys
    if '--real-check' in sys.argv:
        model=load_full(1,168023000)
        print(json.dumps(dict(real_checkpoint_strict_load=True,parameters=sum(p.numel() for p in model.parameters()))))
        with np.load(BASE/'data/fold_01/biomed/gse176078_brca.npz') as d:
            features=d['features'].copy();patient_ids=d['patient_ids'];test=d['test_idx'];edges=d['edge_index'].copy()
        patient=np.unique(patient_ids[test])[0];ids=test[patient_ids[test]==patient]
        mapping=np.full(len(features),-1,dtype=int);mapping[ids]=np.arange(len(ids))
        keep=(mapping[edges[0]]>=0)&(mapping[edges[1]]>=0)
        with np.load(BASE/f'runs/state_hyper_cl/fold_01/seed_168023000/state_representations/heldout_{patient}.npz') as d:
            prior=d['activity_prior'].copy();reference=d['probability'].copy()
            assert np.array_equal(d['global_idx'],ids)
        with torch.no_grad():
            fused=model.state_module(torch.from_numpy(features[ids]),torch.from_numpy(prior),torch.zeros(len(ids),dtype=torch.long))
            probability=model._forward_task(fused,torch.from_numpy(mapping[edges[:,keep]]).long()).softmax(-1).numpy()
        error=float(np.abs(probability-reference).max())
        assert error<2e-4,('Baseline predictions differ',error)
        from scipy import sparse
        counts=sparse.load_npz(BASE/'data/gse176078_brca_counts.npz')[:8].tocsr()
        genes=np.load(BASE/'data/gse176078_brca_genes.npy')
        projected,recovered_prior,_=project_external(counts,genes,1)
        with np.load(BASE/'state/fold_01/biostate.npz') as d:expected_prior=d['prior'][:8]
        feature_error=float(np.abs(projected-features[:8]).max())
        prior_error=float(np.abs(recovered_prior-expected_prior).max())
        assert feature_error<1e-4 and prior_error<1e-4,(feature_error,prior_error)
        print(json.dumps(dict(real_prediction_max_error=error,source_transform_replay_feature_error=feature_error,
            source_transform_replay_prior_error=prior_error)))
