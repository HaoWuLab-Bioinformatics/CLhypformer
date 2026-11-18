"""Isolated evaluator for nonlinear biological-state hypergraphs.

The original patient evaluator still handles checkpoint selection, predictions
and metrics. This adapter changes only the optional state branch and loss.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from gse_shared_eval import sha256, split_sha


MODES = ('removed', 'encoder', 'linear', 'nonlinear', 'shuffled')


def install(data_path, state_path, mode, temperature, log_path):
    import hypformer
    import gse_inductive_eval as evaluator
    from gse_biostate_model import BiologicalStateHypergraph
    with np.load(data_path, allow_pickle=False) as base, np.load(state_path, allow_pickle=False) as source:
        if source['split_sha256'].item() != split_sha(base) or not np.array_equal(
                base['barcodes'], source['barcodes']):
            raise ValueError('Biological states and fold data are not aligned')
        prior = source['prior'].copy()
        training_prior = prior[base['train_idx']]
        templates = ((training_prior.T @ base['features'][base['train_idx']]) /
                     np.maximum(training_prior.sum(0)[:, None], 1e-6)).astype(np.float32)
        positive = source['train_edge_positive'].copy()
        negative = source['negative_program'].copy()
        names = source['program_names'].copy()
        barcodes = base['barcodes'].copy()
        patients = base['patient_ids'].copy()
        valid_indices = set(map(int, base['valid_idx']))
        if not np.array_equal(source['train_idx'], base['train_idx']):
            raise ValueError('Training cell index order differs')
        codes = np.unique(base['patient_ids'][base['train_idx']].astype(str), return_inverse=True)[1]
        if not np.array_equal(codes, source['train_patient_codes']):
            raise ValueError('Patient-program edge ordering differs')
    old_init = hypformer.HypFormer.__init__
    capture = dict(model=None, datasets={})

    def initialize(self, args):
        old_init(self, args)
        # Build identical parameters in identical order for all ablations.
        self.state_module = BiologicalStateHypergraph(args.in_channels, len(names),
                                                      self.manifold_hidden, mode=mode)
        self.state_module.expression_templates.copy_(torch.from_numpy(templates))
        self.register_buffer('state_positive', torch.from_numpy(positive))
        self.register_buffer('state_negative', torch.from_numpy(negative))
        for unused in (self.adv_module, self.adapt_layer, self.cl_input_norm):
            if unused is not None:
                unused.requires_grad_(False)
        self.cl_gate_logit.requires_grad_(False)
        self._state_step = 0
        capture['model'] = self

    def forward(self, dataset, train_idx=None):
        x = dataset.graph['node_feat']
        idx = np.asarray(dataset.global_idx)
        activity = torch.as_tensor(prior[idx], device=x.device, dtype=x.dtype)
        fused = self.state_module(x, activity, dataset.patient_codes, idx)
        logits = self._forward_task(fused, dataset.graph['edge_index'])
        capture['datasets'][tuple(idx)] = dataset
        if not self.training or train_idx is None:
            return logits
        zero = logits.sum() * 0
        cl = zero
        diagnostics = dict(edge_loss=0.0, cell_loss=0.0, eligible_edges=0, eligible_cells=0)
        if self.cl_weight > 0:
            if self.state_module.mode != 'nonlinear':
                raise ValueError('State transfer loss requires nonlinear hyperedges')
            cl, diagnostics = self.state_module.contrast(
                self.state_positive, self.state_negative, dataset.patient_codes, temperature)
            cl = cl * self.cl_weight
        if self._state_step == 0 or (self._state_step + 1) % 10 == 0:
            task = F.cross_entropy(logits, dataset.label.squeeze(1))
            selected = (self.state_module.anchors, self.state_module.edge_update[0].weight)
            a = torch.autograd.grad(task, selected, retain_graph=True, allow_unused=True)
            b = torch.autograd.grad(cl, selected, retain_graph=True, allow_unused=True)
            a_norm = sum(float(g.detach().square().sum()) for g in a if g is not None) ** 0.5
            b_norm = sum(float(g.detach().square().sum()) for g in b if g is not None) ** 0.5
            dot = sum(float((u.detach() * v.detach()).sum()) for u, v in zip(a, b)
                      if u is not None and v is not None)
            diagnostics.update(epoch=self._state_step + 1, task_grad_norm=a_norm,
                               cl_grad_norm=b_norm, gradient_cosine=dot / max(a_norm * b_norm, 1e-12),
                               weighted_cl=float(cl.detach()), cl_weight=self.cl_weight,
                               fusion_gate=float(torch.sigmoid(self.state_module.gate).detach()),
                               state_gate=float(torch.sigmoid(self.state_module.state_gate).detach()),
                               task_loss=float(task.detach()), curvature_k=float(self.manifold_hidden.k),
                               mode=self.state_module.mode)
            if log_path:
                with Path(log_path).open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(diagnostics) + '\n')
        self._state_step += 1
        return logits, zero, zero, cl

    hypformer.HypFormer.__init__ = initialize
    hypformer.HypFormer.forward = forward
    capture.update(names=names, barcodes=barcodes, patients=patients, valid_indices=valid_indices)
    return capture


def export(capture, output):
    model = capture['model']
    model.eval()
    directory = output / 'state_representations'
    directory.mkdir(exist_ok=True)
    evidence = {'shuffled': [], 'encoder': []}
    # Save source-linked cell memberships and both cell/edge Lorentz coordinates.
    with torch.no_grad():
        for item in list(capture['datasets'].values()):
            idx = np.asarray(item.global_idx)
            training = bool((item.label >= 0).any())
            tag = 'train' if training else 'heldout_' + str(capture['patients'][idx[0]])
            real = model(item).softmax(-1)
            last = model.state_module.last
            cells = last['cells'].clone()
            np.savez_compressed(directory / f'{tag}.npz', global_idx=idx,
                                barcodes=capture['barcodes'][idx], patient_ids=capture['patients'][idx],
                                program_names=capture['names'], probability=real.cpu().numpy(),
                                memberships=last['membership'].cpu().numpy(),
                                activity_prior=last['prior'].cpu().numpy(),
                                cells_lorentz=cells.cpu().numpy(),
                                edges_lorentz=last['edges'].cpu().numpy(),
                                support=last['support'].cpu().numpy())
            if (not training and set(map(int, idx)).issubset(capture['valid_indices'])
                    and model.state_module.mode == 'nonlinear'):
                for alternate in evidence:
                    model.state_module.mode = alternate
                    changed = model(item).softmax(-1)
                    new_cells = model.state_module.last['cells']
                    evidence[alternate].append(dict(cells=len(idx),
                        mean_absolute_probability_change=float((real - changed).abs().mean()),
                        prediction_change_fraction=float((real.argmax(1) != changed.argmax(1)).float().mean()),
                        relative_representation_change=float((cells - new_cells).norm() / cells.norm().clamp_min(1e-8))))
                model.state_module.mode = 'nonlinear'
    # Mechanism checks use validation features only, never test predictions.
    result = {}
    for alternate, records in evidence.items():
        if records:
            n = sum(r['cells'] for r in records)
            result[alternate] = {key: sum(r[key] * r['cells'] for r in records) / n
                                 for key in records[0] if key != 'cells'}
    (output / 'mechanism.json').write_text(json.dumps(dict(interventions=result,
        heldout_labels_used=False, scope='validation feature graphs only'), indent=2))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--state-data', type=Path, required=True)
    parser.add_argument('--state-mode', choices=MODES, required=True)
    parser.add_argument('--state-temperature', type=float, default=0.5)
    selected, rest = parser.parse_known_args()
    data_path = Path(rest[rest.index('--npz') + 1])
    output = Path(rest[rest.index('--output-dir') + 1])
    output.mkdir(parents=True, exist_ok=True)
    capture = install(data_path, selected.state_data, selected.state_mode,
                      selected.state_temperature, output / 'state_training.jsonl')
    import gse_inductive_eval as evaluator
    sys.argv = [sys.argv[0], *rest]
    evaluator.main()
    export(capture, output)
    path = output / 'metrics.json'
    metrics = json.loads(path.read_text())
    metrics.update(state_mode=selected.state_mode, state_temperature=selected.state_temperature,
                   state_data_sha256=sha256(selected.state_data), ordinary_graph='patient_mutual_k5',
                   state_evaluator_sha256=sha256(Path(__file__)),
                   state_model_sha256=sha256(Path(__file__).with_name('gse_biostate_model.py')),
                   states_use_cell_type_labels=False, score_method='train-standardized Hallmark expression')
    path.write_text(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
