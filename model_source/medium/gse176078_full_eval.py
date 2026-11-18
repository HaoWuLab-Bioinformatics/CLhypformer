"""Same model/trainer, precomputed sparse patient graphs, full-cohort telemetry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, f1_score, roc_auc_score
from gse_shared_eval import sha256


def install_cached_graphs(path):
    import gse_cohort_inductive_eval as evaluator
    import gse_patient_state_graph as graph_module
    with np.load(path) as source:
        features, patients, edges = source['features'], source['patient_ids'], source['edge_index']
        train, valid, test = (source[k] for k in ('train_idx', 'valid_idx', 'test_idx'))
    assert len(features) == 100064 and np.all(patients[edges[0]] == patients[edges[1]])
    def fingerprint(x, groups):
        value = hashlib.sha256(np.ascontiguousarray(x).tobytes())
        value.update(np.asarray(groups).astype(str).tobytes())
        return value.hexdigest()
    cache = {}
    def subset(ids):
        mapping = np.full(len(features), -1, dtype=np.int64)
        mapping[ids] = np.arange(len(ids))
        keep = (mapping[edges[0]] >= 0) & (mapping[edges[1]] >= 0)
        local = mapping[edges[:, keep]]
        cache[fingerprint(features[ids], patients[ids])] = (local,
            dict(cached=True, cells=len(ids), directed_edges=local.shape[1], cross_patient_edges=0,
                 patients=len(np.unique(patients[ids])), neighbors=5, mutual_neighbours=True))
    subset(train)
    for indices in (valid,test):
        for patient in np.unique(patients[indices]):
            subset(indices[patients[indices] == patient])
    def cached(x, groups, k=5):
        if k != 5:
            raise ValueError('Frozen graph uses k=5')
        return cache[fingerprint(x, groups)]
    def make(features, label, minor, patients, global_idx, device, original):
        local, _ = cached(features, patients)
        edge = torch.from_numpy(local).long().to(device)
        classes, minor_code = np.unique(minor, return_inverse=True)
        if np.all(label < 0):
            minor_code = np.full(len(label), -1, dtype=np.int64)
        _, codes = np.unique(patients, return_inverse=True)
        return SimpleNamespace(graph=dict(node_feat=torch.from_numpy(features.astype(np.float32)).to(device),
            edge_index=(edge,) if original else edge, edge_feat=None, num_nodes=len(features)),
            label=torch.from_numpy(label.astype(np.int64))[:,None].to(device),
            cl_fine_labels=torch.from_numpy(minor_code).long().to(device),
            patient_codes=torch.from_numpy(codes).long().to(device), global_idx=global_idx, minor_classes=classes)
    evaluator._make_graph = make
    graph_module.patient_local_edges = cached


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--state-data', type=Path)
    parser.add_argument('--state-mode', choices=('removed','encoder','nonlinear','shuffled'))
    args, rest = parser.parse_known_args()
    data = Path(rest[rest.index('--npz')+1])
    output = Path(rest[rest.index('--output-dir')+1])
    output.mkdir(parents=True, exist_ok=True)
    install_cached_graphs(data)
    capture = None
    if args.state_mode:
        from gse_biostate_eval import install
        capture = install(data, args.state_data, args.state_mode, .5, output / 'state_training.jsonl')
        import hypformer
        forward = hypformer.HypFormer.forward
        stamp = {'epoch':0, 'last':None}
        def observed(self, dataset, train_idx=None):
            begin = time.monotonic()
            if self.training and train_idx is not None:
                stamp['epoch'] += 1
                torch.cuda.reset_peak_memory_stats()
            result = forward(self, dataset, train_idx)
            if self.training and train_idx is not None:
                torch.cuda.synchronize()
                value = dict(epoch=stamp['epoch'], training_cells=len(dataset.global_idx),
                    forward_seconds=time.monotonic()-begin,
                    previous_epoch_seconds=begin-stamp['last'] if stamp['last'] else None,
                    peak_cuda_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    peak_cuda_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
                stamp['last'] = begin
                with (output / 'runtime.jsonl').open('a') as stream:
                    stream.write(json.dumps(value)+'\n')
            return result
        hypformer.HypFormer.forward = observed
        # The full-scale branch must never call the legacy dense N-by-N builder.
        def forbidden(*a, **kw):
            raise RuntimeError('Legacy dense contrastive branch called in full-cohort run')
        hypformer.HypHypergraphAdversarialModule.forward = forbidden
    import gse_cohort_inductive_eval as evaluator
    sys.argv = [sys.argv[0], *rest]
    start = time.monotonic()
    evaluator.main()
    if capture:
        from gse_biostate_eval import export
        export(capture, output)
    path = output / 'metrics.json'
    metrics = json.loads(path.read_text())
    with np.load(data) as source, np.load(output / 'predictions.npz') as prediction:
        truth, pred, probability = prediction['true'], prediction['pred'], prediction['prob']
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, pred, labels=np.arange(9), zero_division=0)
        metrics['test_weighted_f1'] = float(f1_score(truth,pred,average='weighted',zero_division=0))
        metrics['test_macro_precision'] = float(precision.mean())
        metrics['test_balanced_accuracy'] = float(recall[support>0].mean())
        metrics['test_class_metrics'] = {str(name):dict(precision=float(p),recall=float(r),f1=float(f),support=int(n))
            for name,p,r,f,n in zip(source['class_names'],precision,recall,f1,support)}
        aurocs = {}
        for code,name in enumerate(source['class_names']):
            binary = truth == code
            if binary.any() and not binary.all():
                aurocs[str(name)] = float(roc_auc_score(binary,probability[:,code]))
        metrics['test_class_ovr_auroc'] = aurocs
        metrics['test_macro_ovr_auroc_observed_classes'] = float(np.mean(list(aurocs.values()))) if aurocs else None
        donors = source['patient_ids'][prediction['test_idx']]
        patient_metrics = []
        for donor in np.unique(donors):
            mask = donors == donor
            pr,rr,ff,nn = precision_recall_fscore_support(truth[mask],pred[mask],labels=np.arange(9),zero_division=0)
            patient_metrics.append(dict(patient=str(donor),cells=int(mask.sum()),
                accuracy=float(np.mean(truth[mask]==pred[mask])),
                macro_f1_observed=float(f1_score(truth[mask],pred[mask],labels=np.unique(truth[mask]),average='macro',zero_division=0)),
                class_support=nn.tolist(),class_recall=rr.tolist()))
        (output / 'per_patient.json').write_text(json.dumps(patient_metrics,indent=2))
    metrics.update(full_cohort_cells=100064, subsampling=False, cached_patient_graphs=True,
        state_mode=args.state_mode, total_seconds=time.monotonic()-start,
        peak_cuda_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        full_adapter_sha256=sha256(Path(__file__)))
    path.write_text(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
