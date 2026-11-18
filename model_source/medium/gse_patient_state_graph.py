"""Label-free patient-local cell graph and multiscale hyperedges.

The same rules apply to pooled training patients and to a new patient at
inference. No class label, clinical subtype, or test split indicator enters
either construction. Expression PCs are the only cell-state coordinates.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch


def patient_local_edges(features, patient_ids, k=5):
    """Mutual local kNN with one-neighbour fallback for isolated cells.

    Unlike retaining only the within-patient part of a global graph, each
    patient gets a fresh local search. Mutuality avoids forcing distant cell
    states into the graph. A cell without any mutual neighbour receives one
    nearest-neighbour edge so it remains connected at inference too.
    """
    x = np.asarray(features, dtype=np.float32)
    patient = np.asarray(patient_ids).astype(str)
    if x.ndim != 2 or len(x) != len(patient) or not np.isfinite(x).all():
        raise ValueError("Patient-local graph requires finite aligned features")
    if k < 1:
        raise ValueError("k must be positive")
    pairs = set()
    counts = {}
    fallback_count = 0
    for name in np.unique(patient):
        ids = np.flatnonzero(patient == name)
        if len(ids) < 2:
            raise ValueError(f"Patient {name} has fewer than two cells")
        _, neighbors = NearestNeighbors(
            n_neighbors=min(k + 1, len(ids))).fit(x[ids]).kneighbors(x[ids])
        choices = [set(map(int, row[row != i]))
                   for i, row in enumerate(neighbors)]
        local_pairs = set()
        for i, row in enumerate(choices):
            for j in row:
                if i in choices[j]:
                    local_pairs.add((min(i, j), max(i, j)))
        degree = np.zeros(len(ids), dtype=np.int64)
        for i, j in local_pairs:
            degree[i] += 1
            degree[j] += 1
        for i in np.flatnonzero(degree == 0):
            if degree[i] > 0:
                continue
            j = next(int(j) for j in neighbors[i] if j != i)
            pair = (min(i, j), max(i, j))
            if pair not in local_pairs:
                local_pairs.add(pair)
                fallback_count += 1
                degree[i] += 1
                degree[j] += 1
        counts[name] = len(local_pairs)
        pairs.update((min(int(ids[i]), int(ids[j])),
                      max(int(ids[i]), int(ids[j]))) for i, j in local_pairs)
    pair_array = np.asarray(sorted(pairs), dtype=np.int64).T
    edge = np.concatenate((pair_array, pair_array[::-1]), axis=1)
    order = np.lexsort((edge[1], edge[0]))
    edge = edge[:, order]
    if edge.min() < 0 or edge.max() >= len(x):
        raise ValueError("Out-of-range patient graph edge")
    if np.any(patient[edge[0]] != patient[edge[1]]):
        raise AssertionError("Patient-local graph has a cross-patient edge")
    return edge, {"patients": len(counts), "cells": len(x),
                  "undirected_edges": int(edge.shape[1] // 2),
                  "edges_by_patient": counts, "neighbors": k,
                  "mutual_neighbours": True,
                  "isolate_fallback_edges": fallback_count,
                  "cross_patient_edges": 0}


@torch.no_grad()
def patient_state_incidence(features, patient_codes, short_k=3, long_k=9,
                            outer_weight=0.35):
    """Two-scale cell-state hyperedges, one centered on every cell.

    Columns are hyperedges and rows are cells. The center and its mutually
    nearest three local cells have weight 1; additional mutually nearest
    cells up to nine have weight 0.35. A center with no mutual matches keeps
    a self-only hyperedge. The matrix is an incidence matrix, not an adjacency
    matrix. Memberships are fixed from expression PCs before optimization.
    """
    if features.ndim != 2:
        raise ValueError("Expected a cells-by-features matrix")
    n = features.size(0)
    codes = torch.as_tensor(patient_codes, device=features.device).reshape(-1)
    if codes.numel() != n or n == 0 or short_k < 1 or long_k < short_k:
        raise ValueError("Invalid patient codes or neighborhood sizes")
    if not (0.0 < outer_weight <= 1.0):
        raise ValueError("outer_weight must lie in (0, 1]")
    incidence = features.new_zeros((1, n, n))
    for code in torch.unique(codes):
        ids = torch.nonzero(codes == code, as_tuple=False).flatten()
        size = ids.numel()
        if size > 1:
            distance = torch.cdist(features[ids].float(), features[ids].float())
            distance.fill_diagonal_(float("inf"))
            nearest = torch.topk(distance, k=min(long_k, size - 1),
                                 largest=False).indices
            rows = torch.arange(size, device=features.device)[:, None].expand_as(nearest)
            directed = torch.zeros((size, size), dtype=torch.bool,
                                   device=features.device)
            directed[rows, nearest] = True
            mutual = directed & directed.T
            allowed = mutual[rows, nearest]
            incidence[0, ids[nearest[allowed]], ids[rows[allowed]]] = outer_weight
            core = nearest[:, :min(short_k, size - 1)]
            core_rows = rows[:, :core.size(1)]
            core_allowed = mutual[core_rows, core]
            incidence[0, ids[core[core_allowed]],
                      ids[core_rows[core_allowed]]] = 1.0
        incidence[0, ids, ids] = 1.0
    if torch.any(incidence.sum(dim=1) <= 0) or torch.any(
            incidence.sum(dim=2) <= 0):
        raise AssertionError("Empty cell or hyperedge in patient incidence")
    return incidence
