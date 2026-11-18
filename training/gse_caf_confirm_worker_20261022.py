"""Convergence-controlled confirmation; config selection uses validation only.

Adapt interfaces only. External implementations are fingerprinted, not edited.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

MEDIUM = Path(__file__).resolve().parent
BASE = Path(os.environ['CLHYP_ATLAS_ROOT']) / 'bio_runs/gse176078_full_20261017'
BASELINES = Path(os.environ['CLHYP_BASELINES_ROOT'])
ROOT = BASE.parent/'gse176078_caf_confirm_20261022'
METHODS = ('mlp', 'gcn', 'gat', 'hgnn', 'fhnn', 'lighthgnn')


def write(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, indent=2, allow_nan=False))


def snapshot(model):
    # The local FHNN compatibility tensor has no deepcopy/new_empty support.
    # Serialize plain detached tensors, then load into the original parameters.
    return {key: value.as_subclass(torch.Tensor).detach().clone()
            for key, value in model.state_dict().items()}


def incidence(edge, n):
    # Exact closed-neighborhood conversion, deduplicated and label independent.
    pairs = np.column_stack([edge, edge[::-1], np.stack([np.arange(n)]*2)])
    return torch.from_numpy(np.unique(pairs.T, axis=0).T.copy()).long()


def block(data, ids, method, device):
    mapping = np.full(len(data['features']), -1, dtype=np.int64)
    mapping[ids] = np.arange(len(ids))
    edges = data['edge_index']
    keep = (mapping[edges[0]] >= 0) & (mapping[edges[1]] >= 0)
    e = mapping[edges[:, keep]]
    x = torch.from_numpy(data['features'][ids]).float().to(device)
    graph = torch.from_numpy(e.copy()).long().to(device)
    if method == 'hgnn':
        graph = incidence(e, len(ids)).to(device)
    if method == 'lighthgnn':
        from dhg import Hypergraph
        inc = incidence(e, len(ids)).numpy()
        members = [[] for _ in ids]
        for v, h in inc.T:
            members[h].append(int(v))
        graph = Hypergraph(len(ids), members).to(device)
    if method == 'fhnn':
        from utils.data_utils import process
        adj = sp.coo_matrix((np.ones(e.shape[1]), (e[0], e[1])), shape=(len(ids), len(ids))).tocsr()
        adj.data[:] = 1.
        graph, _ = process(adj, x.cpu().numpy(), 1, 0)
        graph = graph.to(device)
    y = torch.from_numpy(data['labels'][ids]).long().to(device)
    return x, graph, y


class Ordinary(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        if method == 'mlp':
            self.first, self.last = nn.Linear(50, 128), nn.Linear(128, 9)
        elif method == 'gcn':
            from torch_geometric.nn import GCNConv
            self.first, self.last = GCNConv(50, 128), GCNConv(128, 9)
        else:
            from torch_geometric.nn import GATConv
            self.first = GATConv(50, 32, heads=4, dropout=.5)
            self.last = GATConv(128, 9, heads=1, concat=False, dropout=.5)

    def forward(self, x, graph):
        hidden = self.first(x) if self.method == 'mlp' else self.first(x, graph)
        hidden = F.elu(hidden) if self.method == 'gat' else F.relu(hidden)
        z = F.dropout(hidden, p=.5, training=self.training)
        logits = self.last(z) if self.method == 'mlp' else self.last(z, graph)
        return logits, hidden


def factory(method, device, seed, n):
    if method in ('mlp', 'gcn', 'gat'):
        model = Ordinary(method).to(device)
        return model, lambda x, graph: model(x, graph), .005, 5e-4, None
    if method == 'hgnn':
        sys.path.insert(0, str(BASELINES / 'DHG-Bench-main/dhgbench'))
        from lib_models.HNN.hgnn import HCHA
        model = HCHA(50, 9, SimpleNamespace(All_num_layers=2, dropout=.5,
                         MLP_hidden=128, HCHA_symdegnorm=True)).to(device)
        def forward(x, graph):
            hidden, _ = model.convs[0](x, graph)
            hidden = F.elu(hidden)
            logits, _ = model.convs[1](F.dropout(hidden, .5, model.training), graph)
            return logits, hidden
        return model, forward, .001, 0., None
    if method == 'lighthgnn':
        sys.path.insert(0, str(BASELINES / 'LightHGNN-main'))
        from my_model import MyHGNN
        teacher = MyHGNN(50, 32, 9, use_bn=False).to(device)
        return teacher, lambda x, graph: (teacher(x, graph), teacher(x, graph, get_emb=True)), .01, 5e-4, None
    # Isolated worker avoids collisions with this project's manifolds/config.
    repo = BASELINES / 'fully-hyperbolic-nn-main/gcn'
    sys.path.insert(0, str(repo)); sys.path.insert(1, str(repo.parent))
    from config import parser
    from models.base_models import NCModel
    from optim import RiemannianAdam
    cfg = parser.parse_args([])
    cfg.task, cfg.model, cfg.manifold = 'nc', 'HyboNet', 'Lorentz'
    cfg.dataset, cfg.seed, cfg.split_seed = 'bio', seed, seed
    cfg.cuda = 0 if device.type == 'cuda' else -1
    cfg.device = str(device)
    cfg.dim, cfg.num_layers = 16, 4
    cfg.dropout, cfg.lr, cfg.weight_decay = .1, .005, 0.
    cfg.margin, cfg.grad_clip = 2., .5
    cfg.n_nodes, cfg.feat_dim, cfg.n_classes = n, 50, 9
    model = NCModel(cfg).to(device)
    # Repository's compatibility manifolds are not torch Modules/buffers.
    for item in [model, *model.modules()]:
        for value in vars(item).values():
            if hasattr(value, 'k') and torch.is_tensor(value.k):
                value.k = value.k.to(device)
    def forward(x, graph):
        z = model.encode(x, graph)
        logits = model.decoder.decode(z, graph)
        tangent = model.manifold.logmap0(z)[..., 1:]
        if not torch.isfinite(tangent).all():
            raise FloatingPointError('Nonfinite Lorentz logmap0')
        return logits, tangent
    optimizer = RiemannianAdam(model.parameters(), lr=.005, stabilize=1)
    return model, forward, .005, 0., optimizer


def export(args):
    """Read held-out data only after the validation-selected config is frozen."""
    torch.set_num_threads(4)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    output = args.output_dir
    if (output/'export_complete.json').exists(): return
    selection = json.loads((output/'selection.json').read_text())
    if args.method == 'fhnn':
        repo = BASELINES / 'fully-hyperbolic-nn-main/gcn'
        sys.path.insert(0, str(repo)); sys.path.insert(1, str(repo.parent))
    with np.load(BASE/f'data/fold_{args.fold:02d}/biomed/gse176078_brca.npz') as source:
        data = {key:source[key].copy() for key in source.files}
    train, valid, test = [data[k+'_idx'] for k in ('train', 'valid', 'test')]
    model, forward, _, _, _ = factory(args.method, device, args.seed, len(train))
    if args.method == 'lighthgnn':
        from my_model import MyMLPs
        model = MyMLPs(50, 128, 9).to(device)
        forward = lambda x, graph: (model(x), model(x, get_emb=True))
    checkpoint = torch.load(selection['checkpoint'], map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=True); model.eval()
    records = []
    replay = None
    for role, ids in enumerate((train, valid, test)):
        x, graph, _ = block(data, ids, args.method, device)
        with torch.inference_mode(): logits, z = forward(x, graph)
        assert torch.isfinite(logits).all() and torch.isfinite(z).all()
        caf = data['class_names'][data['labels'][ids]].astype(str) == 'CAFs'
        records.append((ids[caf], z.cpu().numpy()[caf], np.full(caf.sum(), role, dtype=np.int8)))
        if role:
            pred = logits.argmax(-1).cpu().numpy()
            np.savez_compressed(output/('validation.npz' if role == 1 else 'test.npz'),
                idx=ids, true=data['labels'][ids], pred=pred, scores=logits.cpu().numpy())
            if role == 1:
                replay = float(f1_score(data['labels'][ids], pred, labels=np.arange(9), average='macro', zero_division=0))
                if abs(replay-selection['validation_macro_f1']) > 1e-6:
                    raise ValueError(('Validation checkpoint replay mismatch', replay, selection['validation_macro_f1']))
        del x, graph, logits, z
    np.savez_compressed(output/'caf.npz', idx=np.concatenate([x[0] for x in records]),
        points=np.concatenate([x[1] for x in records]), role=np.concatenate([x[2] for x in records]))
    del data
    for path in sorted((ROOT/'prepared_external').glob(f'*/fold_{args.fold:02d}.npz')):
        with np.load(path) as source:
            external = {key:source[key].copy() for key in source.files}
        caf = external['caf_idx']
        positions = np.full(len(external['features']), -1, dtype=np.int64)
        positions[caf] = np.arange(len(caf))
        points = None
        for sample in np.unique(external['patient_ids']):
            ids = np.flatnonzero(external['patient_ids'] == sample)
            keep = positions[ids] >= 0
            if not keep.any(): continue
            x, graph, _ = block(external, ids, args.method, device)
            with torch.inference_mode(): _, z = forward(x, graph)
            values = z.cpu().numpy()
            if points is None:
                points = np.full((len(caf), values.shape[1]), np.nan, dtype=np.float32)
            points[positions[ids[keep]]] = values[keep]
            del x, graph, z
        assert points is not None and np.isfinite(points).all()
        np.savez_compressed(output/f'external_{path.parent.name}.npz',
            points=points, barcodes=external['barcodes'], patients=external['patients'])
    write(output/'export_complete.json', dict(complete=True, method=args.method,
        validation_replay_macro_f1=replay, representation_dimensions=records[0][1].shape[1],
        checkpoint=selection['checkpoint'], checkpoint_sha256=hashlib.sha256(Path(selection['checkpoint']).read_bytes()).hexdigest(),
        external_fit=False))


def run(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output/'complete.json').exists():
        return
    with np.load(BASE/f'data/fold_{args.fold:02d}/biomed/gse176078_brca.npz') as source:
        data = {k: source[k].copy() for k in source.files}
    if args.smoke:
        # Label-stratified subsets, all three roles remain disjoint.
        for role in ('train', 'valid', 'test'):
            all_ids = data[role+'_idx']
            data[role+'_idx'] = np.concatenate([all_ids[data['labels'][all_ids] == c][:12] for c in range(9)])
    if args.method == 'fhnn':
        repo = BASELINES / 'fully-hyperbolic-nn-main/gcn'
        sys.path.insert(0, str(repo)); sys.path.insert(1, str(repo.parent))
    train, valid, test = [data[key+'_idx'] for key in ('train', 'valid', 'test')]
    for left, right in ((train, valid), (train, test), (valid, test)):
        assert not set(data['patient_ids'][left]) & set(data['patient_ids'][right])
    tr = block(data, train, args.method, device)
    va = block(data, valid, args.method, device)
    # Ordinary graphs have no cross-patient edges; HGNN conversion preserves this.
    e = data['edge_index']
    assert np.all(data['patient_ids'][e[0]] == data['patient_ids'][e[1]])
    model, forward, lr, wd, optimizer = factory(args.method, device, args.seed, len(train))
    lr *= args.lr_scale
    if optimizer is not None:
        for group in optimizer.param_groups: group['lr'] = lr
    epochs = 3 if args.smoke else args.max_epochs
    minimum = 1 if args.smoke else args.min_epochs
    history = []

    def fit(current, callback, phase, teacher=None, optim=None):
        optim = optim or torch.optim.Adam(current.parameters(), lr=lr, weight_decay=wd)
        best, saved, best_epoch, stale = -1., None, 0, 0
        for epoch in range(1, epochs+1):
            current.train(); optim.zero_grad()
            logits, _ = callback(tr[0], tr[1])
            if teacher is not None:
                loss = F.kl_div(F.log_softmax(logits, -1), teacher, reduction='batchmean', log_target=False)
            elif args.method == 'fhnn':
                positive = logits.gather(1, tr[2][:, None])
                mask = ~F.one_hot(tr[2], 9).bool()
                loss = F.relu(2. - positive + logits)[mask].mean()
            else:
                loss = F.cross_entropy(logits, tr[2])
            if not torch.isfinite(loss):
                raise FloatingPointError((phase, epoch, 'loss'))
            loss.backward()
            if args.method == 'fhnn': torch.nn.utils.clip_grad_norm_(current.parameters(), .5)
            optim.step(); current.eval()
            with torch.no_grad():
                pred, _ = callback(va[0], va[1])
            score = float(f1_score(va[2].cpu(), pred.argmax(-1).cpu(), labels=np.arange(9), average='macro', zero_division=0))
            if score > best:
                best, saved, best_epoch = score, snapshot(current), epoch
                stale = 0
            else:
                stale += 1
            history.append(dict(phase=phase, epoch=epoch, loss=float(loss.detach()), validation_macro_f1=score))
            if epoch % 10 == 0 or args.smoke:
                print(json.dumps(history[-1]), flush=True)
            if epoch >= minimum and stale >= args.patience:
                break
        current.load_state_dict(saved); current.eval()
        return best_epoch, best

    best_epoch, validation = fit(model, forward, 'teacher' if args.method == 'lighthgnn' else 'classifier', optim=optimizer)
    if args.method == 'lighthgnn':
        with torch.no_grad():
            target = forward(tr[0], tr[1])[0].softmax(-1).detach()
        teacher_state = snapshot(model)
        from my_model import MyMLPs
        model = MyMLPs(50, 128, 9).to(device)
        forward = lambda x, graph: (model(x), model(x, get_emb=True))
        best_epoch, validation = fit(model, forward, 'student', teacher=target)
        torch.save(teacher_state, output/'teacher.pt')
    torch.save(dict(state_dict=snapshot(model), method=args.method), output/'checkpoint.pt')
    write(output/'history.json', history)
    if args.train_only:
        write(output/'complete.json', dict(method=args.method, fold=args.fold, seed=args.seed,
            validation_macro_f1=validation, best_epoch=best_epoch, learning_rate=lr,
            lr_scale=args.lr_scale, max_epochs=epochs, min_epochs=minimum, patience=args.patience,
            actual_epochs_per_stage={phase:len([x for x in history if x['phase']==phase])
                                    for phase in set(x['phase'] for x in history)},
            selected_using='validation_macro_f1_only', test_exported=False))
        return
    records = []
    for role, ids in enumerate((train, valid, test)):
        if role == 0: current = tr
        elif role == 1: current = va
        else: current = block(data, ids, args.method, device)
        with torch.no_grad():
            logits, z = forward(current[0], current[1])
        assert torch.isfinite(logits).all() and torch.isfinite(z).all()
        z = z.cpu().numpy()
        caf = data['class_names'][data['labels'][ids]].astype(str) == 'CAFs'
        records.append((ids[caf], z[caf], np.full(caf.sum(), role, dtype=np.int8)))
        if role:
            np.savez_compressed(output/('validation.npz' if role == 1 else 'test.npz'),
                idx=ids, true=data['labels'][ids], pred=logits.argmax(-1).cpu().numpy(),
                scores=logits.cpu().numpy())
    np.savez_compressed(output/'caf.npz', idx=np.concatenate([x[0] for x in records]),
        points=np.concatenate([x[1] for x in records]), role=np.concatenate([x[2] for x in records]))
    write(output/'history.json', history)
    external = {'fhnn':str(BASELINES / 'fully-hyperbolic-nn-main/gcn/models/base_models.py'),
                'hgnn':str(BASELINES / 'DHG-Bench-main/dhgbench/lib_models/HNN/hgnn.py'),
                'lighthgnn':str(BASELINES / 'LightHGNN-main/my_model.py')}.get(args.method)
    sources = {str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if external:
        sources[external] = hashlib.sha256(Path(external).read_bytes()).hexdigest()
    write(output/'complete.json', dict(method=args.method, fold=args.fold, seed=args.seed,
        cells=len(train)+len(valid)+len(test), input_dimensions=50,
        representation_dimensions=z.shape[1], best_epoch=best_epoch, validation_macro_f1=validation,
        epochs_per_stage=epochs, learning_rate=lr, weight_decay=wd, source_sha256=sources,
        selected_using='validation_macro_f1_only', test_used_for_training=False,
        representation='Lorentz logmap0 spatial coordinates' if args.method == 'fhnn' else 'penultimate hidden activation',
        caveat='External repositories are existing local implementations, not certified pristine upstream releases'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--seed', type=int, default=168023000)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--lr-scale', type=float, choices=(.2, 1.), default=1.)
    parser.add_argument('--max-epochs', type=int, default=400)
    parser.add_argument('--min-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=75)
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    export(args) if args.export_only else run(args)
