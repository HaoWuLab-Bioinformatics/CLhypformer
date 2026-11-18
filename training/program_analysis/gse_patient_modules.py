"""Experimental patient-aware graph and cross-patient prototype objective.

These components are isolated from the frozen ten-seed confirmation code.
Both use only expression features and training-patient labels as specified;
validation and test labels must never be passed into the prototype loss.
"""

from collections import Counter
from itertools import combinations

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn.functional as F


def patient_aware_edges(features, patient_ids, base_k=10, cross_k=2,
                        max_cross_per_cell=3):
    """Filter a global kNN graph using patient-aware cross-patient MNN anchors.

    The same-patient edges of the global graph are retained. Cross-patient
    edges must also be mutual nearest neighbours between the two patients;
    each cell is capped at ``max_cross_per_cell`` such anchors. This avoids
    forcing k neighbours from a small patient into a heterogeneous cell type.
    Labels, clinical subtypes, and split membership are never used here.
    """
    x = np.asarray(features, dtype=np.float32)
    patient_ids = np.asarray(patient_ids).astype(str)
    if x.ndim != 2 or len(x) != len(patient_ids):
        raise ValueError("features/patient_ids must align as cells by features")
    if not np.isfinite(x).all():
        raise ValueError("non-finite feature values")
    if min(base_k, cross_k, max_cross_per_cell) < 1:
        raise ValueError("neighbour counts must be positive")
    groups = {p: np.flatnonzero(patient_ids == p)
              for p in np.unique(patient_ids)}
    base = set()
    if len(x) > 1:
        near = NearestNeighbors(n_neighbors=min(base_k + 1, len(x)))
        near.fit(x)
        for i, options in enumerate(near.kneighbors(x, return_distance=False)):
            for j in options:
                if i != j:
                    base.add((min(i, int(j)), max(i, int(j))))
    local = {pair for pair in base if patient_ids[pair[0]] == patient_ids[pair[1]]}
    base_cross = base - local

    candidates = []
    for pa, pb in combinations(groups, 2):
        a, b = groups[pa], groups[pb]
        ka, kb = min(cross_k, len(b)), min(cross_k, len(a))
        ab_dist, ab_idx = NearestNeighbors(n_neighbors=ka).fit(x[b]).kneighbors(x[a])
        _, ba_idx = NearestNeighbors(n_neighbors=kb).fit(x[a]).kneighbors(x[b])
        for ai in range(len(a)):
            for rank, bj in enumerate(ab_idx[ai]):
                if ai in ba_idx[bj]:
                    i, j = int(a[ai]), int(b[bj])
                    pair = (min(i, j), max(i, j))
                    if pair in base_cross:
                        candidates.append((float(ab_dist[ai, rank]), *pair))
    candidates.sort()
    degree = np.zeros(len(x), dtype=np.int32)
    cross = set()
    for _, i, j in candidates:
        if (i, j) not in cross and degree[i] < max_cross_per_cell and degree[j] < max_cross_per_cell:
            cross.add((i, j))
            degree[i] += 1
            degree[j] += 1
    pairs = sorted(local) + sorted(cross)
    if pairs:
        forward = np.asarray(pairs, dtype=np.int64).T
        edges = np.concatenate((forward, forward[::-1]), axis=1)
        pair_kind = np.asarray([0] * len(local) + [1] * len(cross), dtype=np.int8)
        kind = np.concatenate((pair_kind, pair_kind))
    else:
        edges = np.empty((2, 0), dtype=np.int64)
        kind = np.empty(0, dtype=np.int8)
    diagnostics = {"cells": len(x), "patients": len(groups),
                   "base_undirected_edges": len(base),
                   "base_cross_undirected_edges": len(base_cross),
                   "local_undirected_edges": len(local),
                   "cross_undirected_edges": len(cross),
                   "cross_degree_max": int(degree.max(initial=0)),
                   "cross_connected_cells": int(np.count_nonzero(degree)),
                   "cross_edges_per_patient_pair": dict(Counter(
                       "|".join(sorted((patient_ids[i], patient_ids[j])))
                       for i, j in cross))}
    return edges, kind, diagnostics


def cross_patient_prototype_loss(embeddings, labels, patient_codes, train_idx,
                                 temperature=0.2, state_labels=None,
                                 expression_view=None, hyper_view=None):
    """Class-balanced prototype CE with leave-one-patient-out positives.

    The positive prototype for an anchor contains only training cells of the
    same major class from other patients. Other major classes are negatives.
    All prototypes are detached, preventing self-target shortcuts. A class
    seen in only one training patient contributes no anchors.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    ids = torch.as_tensor(train_idx, dtype=torch.long, device=embeddings.device)
    y = torch.as_tensor(labels, dtype=torch.long, device=embeddings.device)[ids]
    patient = torch.as_tensor(patient_codes, dtype=torch.long,
                              device=embeddings.device)[ids]
    if bool((y < 0).any()) or bool((patient < 0).any()):
        raise ValueError("Prototype loss received an unlabeled or unknown training cell")
    z = F.normalize(embeddings[ids], dim=-1)
    if not len(ids):
        return embeddings.sum() * 0, {"eligible_anchors": 0, "eligible_classes": 0}
    classes = torch.unique(y, sorted=True)
    if len(classes) < 2:
        return embeddings.sum() * 0, {"eligible_anchors": 0, "eligible_classes": 0}
    # Each patient contributes one centroid, so a large biopsy cannot
    # dominate the target of another patient's cells.
    with torch.no_grad():
        centers = []
        for cell_class in classes:
            members = y == cell_class
            patient_centers = torch.stack([
                z[members & (patient == p)].mean(0)
                for p in torch.unique(patient[members])
            ])
            centers.append(F.normalize(patient_centers.mean(0), dim=0))
        centers = torch.stack(centers)
    class_losses = []
    eligible_anchors = 0
    for column, cell_class in enumerate(classes):
        class_mask = y == cell_class
        if torch.unique(patient[class_mask]).numel() < 2:
            continue
        within_class = []
        for patient_code in torch.unique(patient[class_mask]):
            anchors = class_mask & (patient == patient_code)
            others = class_mask & (patient != patient_code)
            if not bool(others.any()):
                continue
            with torch.no_grad():
                other_patients = torch.unique(patient[others])
                positive = F.normalize(torch.stack([
                    z[class_mask & (patient == other)].mean(0)
                    for other in other_patients
                ]).mean(0), dim=0)
            logits = z[anchors] @ centers.T / temperature
            logits = logits.clone()
            logits[:, column] = z[anchors] @ positive / temperature
            target = torch.full((int(anchors.sum()),), column, dtype=torch.long,
                                device=embeddings.device)
            within_class.append(F.cross_entropy(logits, target))
            eligible_anchors += int(anchors.sum())
        if within_class:
            class_losses.append(torch.stack(within_class).mean())
    if not class_losses:
        return embeddings.sum() * 0, {"eligible_anchors": 0, "eligible_classes": 0}
    return torch.stack(class_losses).mean(), {
        "eligible_anchors": eligible_anchors,
        "eligible_classes": len(class_losses),
    }


def prototype_gradient_diagnostics(task_loss, prototype_branch_loss, model):
    """Measure both objectives where prototype gradients reach prediction.

    The old SupCon diagnostic only watched the hypergraph encoder. A prototype
    defined after task fusion also updates the transformer and graph backbones.
    The top-level ratio includes all three disjoint groups below.
    """
    groups = {
        "hypergraph_encoder": tuple(model.adv_module.encoder_l1.parameters())
                              + tuple(model.adv_module.encoder_l2.parameters()),
        "fusion_adapter": tuple(model.adapt_layer.parameters())
                          + tuple(model.cl_input_norm.parameters())
                          + (model.cl_gate_logit,),
        "predictive_backbone": tuple(model.trans_conv.parameters())
                               + tuple(model.graph_conv.parameters()),
    }
    result = {}
    total_task_sq = total_cl_sq = total_dot = 0.0
    for name, parameters in groups.items():
        parameters = tuple(p for p in parameters if p.requires_grad)
        task = torch.autograd.grad(task_loss, parameters, retain_graph=True,
                                   allow_unused=True)
        contrast = torch.autograd.grad(prototype_branch_loss, parameters,
                                       retain_graph=True, allow_unused=True)
        task_sq = sum(float((g.detach() ** 2).sum()) for g in task if g is not None)
        cl_sq = sum(float((g.detach() ** 2).sum()) for g in contrast if g is not None)
        dot = sum(float((a.detach() * b.detach()).sum())
                  for a, b in zip(task, contrast) if a is not None and b is not None)
        task_norm, cl_norm = task_sq ** 0.5, cl_sq ** 0.5
        result[name] = {
            "task_grad_norm": task_norm,
            "cl_grad_norm": cl_norm,
            "cl_to_task_grad_ratio": cl_norm / max(task_norm, 1e-12),
            "grad_cosine": dot / max(task_norm * cl_norm, 1e-12),
        }
        total_task_sq += task_sq
        total_cl_sq += cl_sq
        total_dot += dot
    task_norm, cl_norm = total_task_sq ** 0.5, total_cl_sq ** 0.5
    result.update({
        "task_grad_norm": task_norm,
        "cl_grad_norm": cl_norm,
        "cl_to_task_grad_ratio": cl_norm / max(task_norm, 1e-12),
        "grad_cosine": total_dot / max(task_norm * cl_norm, 1e-12),
    })
    return result


def apply_prototype_pcgrad(task_loss, prototype_branch_loss, model):
    """Preserve the task gradient and project a conflicting prototype gradient.

    Call after optimizer.zero_grad() and before optimizer.step(). Only the
    parameters shared by the classifier and prototype objective are replaced;
    task-only decoder gradients retain the ordinary backward result.
    """
    parameters = tuple(p for p in (
        tuple(model.adv_module.encoder_l1.parameters())
        + tuple(model.adv_module.encoder_l2.parameters())
        + tuple(model.adapt_layer.parameters())
        + tuple(model.cl_input_norm.parameters())
        + (model.cl_gate_logit,)
        + tuple(model.trans_conv.parameters())
        + tuple(model.graph_conv.parameters())) if p.requires_grad)
    task = torch.autograd.grad(task_loss, parameters, retain_graph=True,
                               allow_unused=True)
    contrast = torch.autograd.grad(prototype_branch_loss, parameters,
                                   retain_graph=True, allow_unused=True)
    dot = sum(float((a.detach() * b.detach()).sum())
              for a, b in zip(task, contrast) if a is not None and b is not None)
    task_sq = sum(float((g.detach() ** 2).sum()) for g in task if g is not None)
    (task_loss + prototype_branch_loss).backward()
    applied = dot < 0 and task_sq > 0
    if applied:
        coefficient = dot / task_sq
        for parameter, task_gradient, contrast_gradient in zip(
                parameters, task, contrast):
            if task_gradient is not None and contrast_gradient is not None:
                parameter.grad = (task_gradient + contrast_gradient -
                                  coefficient * task_gradient).detach()
    return {"pcgrad_applied": bool(applied), "pcgrad_preprojection_dot": dot,
            "pcgrad_task_grad_sq": task_sq}
