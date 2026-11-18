"""Isolated GSE176078 gene-hyperedge and cell-state contrast ablations.

The NPZ is fitted using training patients only. Validation and test labels are
never used to build hyperedges or contrastive pairs. This module is activated
only when GSE_BIO_MODE=1, leaving earlier experiment entry points intact.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HYPER_VARIANTS = ("branch_removed", "program_overlap",
                  "program_overlap_shuffled", "gene_expression",
                  "gene_expression_shuffled", "state_local",
                  "state_local_shuffled", "state_local_centered",
                  "state_local_centered_shuffled")
CL_VARIANTS = ("minor_prototype", "cross_view", "state_mnn")


def _training_labels(embeddings, labels, patient_codes, train_idx, state_labels):
    ids = torch.as_tensor(train_idx, dtype=torch.long, device=embeddings.device)
    major = torch.as_tensor(labels, dtype=torch.long, device=embeddings.device)[ids]
    patients = torch.as_tensor(patient_codes, dtype=torch.long,
                               device=embeddings.device)[ids]
    if state_labels is None:
        raise ValueError("Training-only cell-state labels are required")
    states = torch.as_tensor(state_labels, dtype=torch.long,
                             device=embeddings.device)[ids]
    if not len(ids) or bool((major < 0).any() or (states < 0).any() or
                            (patients < 0).any()):
        raise ValueError("A training anchor has a missing class, state, or patient")
    return ids, major, states, patients


def minor_prototype_loss(embeddings, labels, patient_codes, train_idx,
                         temperature=0.2, state_labels=None,
                         expression_view=None, hyper_view=None):
    """Fine-state positives from other training patients, with state negatives.

    States occurring in just one training patient are skipped; no fallback to
    their nine-class average is made. Major classes contribute equal weight.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    ids, major, states, patient = _training_labels(
        embeddings, labels, patient_codes, train_idx, state_labels)
    z = F.normalize(embeddings[ids], dim=-1)
    unique_states = torch.unique(states, sorted=True)
    with torch.no_grad():
        centers = torch.stack([
            F.normalize(z[states == s].mean(0), dim=0) for s in unique_states])
    class_losses = {}
    eligible, eligible_states = 0, 0
    for s_idx, state in enumerate(unique_states):
        mask = states == state
        state_patients = torch.unique(patient[mask], sorted=True)
        if len(state_patients) < 2:
            continue
        state_major = torch.unique(major[mask])
        if len(state_major) != 1:
            raise ValueError("A fine state spans multiple major classes")
        eligible_states += 1
        negative_centers = centers[unique_states != state]
        if not len(negative_centers):
            continue
        per_patient = []
        for p in state_patients:
            anchor = mask & (patient == p)
            other = mask & (patient != p)
            with torch.no_grad():
                # Equal patient weighting avoids making a large tumor the target.
                positive = F.normalize(torch.stack([
                    z[other & (patient == q)].mean(0)
                    for q in state_patients if bool(q != p)
                ]).mean(0), dim=0)
            logits = torch.cat((z[anchor] @ positive[:, None],
                                z[anchor] @ negative_centers.T), dim=1)
            targets = torch.zeros(logits.size(0), dtype=torch.long,
                                  device=z.device)
            per_patient.append(F.cross_entropy(logits / temperature, targets))
            eligible += int(anchor.sum())
        key = int(state_major.item())
        class_losses.setdefault(key, []).append(torch.stack(per_patient).mean())
    if not class_losses:
        return embeddings.sum() * 0, {"eligible_anchors": 0,
                                       "eligible_states": 0,
                                       "eligible_classes": 0}
    loss = torch.stack([
        torch.stack(state_losses).mean() for state_losses in class_losses.values()
    ]).mean()
    return loss, {"eligible_anchors": eligible,
                  "eligible_states": eligible_states,
                  "eligible_classes": len(class_losses)}


def cross_view_loss(embeddings, labels, patient_codes, train_idx,
                    temperature=0.2, state_labels=None,
                    expression_view=None, hyper_view=None):
    """Expression vs gene-hypergraph contrast with masked false negatives.

    Each anchor's two views are its positive pair. Other cells of the same
    fine state are excluded from negatives; nearest remaining cells act as
    hard negatives. The balanced minibatch uses training patients only.
    """
    if expression_view is None or hyper_view is None or temperature <= 0:
        raise ValueError("Cross-view loss requires two aligned views")
    ids, major, states, patient = _training_labels(
        embeddings, labels, patient_codes, train_idx, state_labels)
    if expression_view.shape != hyper_view.shape or len(expression_view) != len(embeddings):
        raise ValueError("Expression and hypergraph views must align by cell")
    # Cap pairwise logits while ensuring every training patient contributes.
    groups = torch.unique(patient, sorted=True)
    per_patient = max(1, 256 // len(groups))
    chosen = []
    for p in groups:
        local = torch.nonzero(patient == p, as_tuple=False).flatten()
        chosen.append(local[torch.randperm(len(local), device=local.device)
                            [:per_patient]])
    batch = torch.cat(chosen)
    if len(batch) < 2:
        return embeddings.sum() * 0, {"eligible_anchors": 0,
                                       "hard_negatives": 0}
    x = F.normalize(expression_view[ids[batch]].detach(), dim=-1)
    h = F.normalize(hyper_view[ids[batch]], dim=-1)
    similarities = x @ h.T / temperature
    not_same_state = states[batch, None] != states[batch][None, :]
    not_same_state.fill_diagonal_(False)
    neg = similarities.masked_fill(~not_same_state, float("-inf"))
    k = min(32, len(batch) - 1)
    hard = neg.topk(k, dim=1).values
    valid = torch.isfinite(hard[:, 0])
    if not bool(valid.any()):
        return hyper_view.sum() * 0, {"eligible_anchors": 0,
                                      "hard_negatives": 0}
    positive = similarities.diagonal()[valid, None]
    logits = torch.cat((positive, hard[valid]), dim=1)
    loss = F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long,
                                               device=logits.device))
    return loss, {"eligible_anchors": int(valid.sum()),
                  "hard_negatives": k, "patients_in_batch": len(groups)}


def state_mnn_loss(embeddings, labels, patient_codes, train_idx,
                   temperature=0.2, state_labels=None,
                   expression_view=None, hyper_view=None):
    """Cross-patient mutual-neighbor positives on the task representation.

    Training-only fine-state labels screen positives and avoid false negatives.
    Each major class contributes equally, and same-major different-state cells
    provide hard negatives. No validation or test cell enters the mining pool.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    ids, major, states, patient = _training_labels(
        embeddings, labels, patient_codes, train_idx, state_labels)
    z = F.normalize(embeddings[ids], dim=-1)
    with torch.no_grad():
        similarity = z.detach() @ z.detach().T
        valid_positive = ((states[:, None] == states[None, :]) &
                          (patient[:, None] != patient[None, :]))
        candidates = similarity.masked_fill(~valid_positive, float("-inf"))
        neighbor_count = min(3, len(ids))
        neighbor_scores, neighbor_ids = candidates.topk(neighbor_count, dim=1)
        neighbor_ok = torch.isfinite(neighbor_scores)
        reciprocal = torch.zeros_like(valid_positive)
        source = torch.arange(len(ids), device=z.device)[:, None].expand_as(neighbor_ids)
        reciprocal[source[neighbor_ok], neighbor_ids[neighbor_ok]] = True
        reciprocal = reciprocal & reciprocal.T
        chosen = []
        for label in torch.unique(major):
            cells = torch.nonzero(major == label, as_tuple=False).flatten()
            cells = cells[torch.randperm(len(cells), device=z.device)[:32]]
            chosen.extend(cells.tolist())
    per_class = {}
    eligible = 0
    for anchor in chosen:
        positives = torch.nonzero(reciprocal[anchor], as_tuple=False).flatten()
        if not len(positives):
            continue
        with torch.no_grad():
            hard_pool = torch.nonzero((major == major[anchor]) &
                                      (states != states[anchor]),
                                      as_tuple=False).flatten()
            easy_pool = torch.nonzero(major != major[anchor],
                                      as_tuple=False).flatten()
            hard = hard_pool[similarity[anchor, hard_pool].topk(
                min(16, len(hard_pool))).indices] if len(hard_pool) else hard_pool
            easy = easy_pool[similarity[anchor, easy_pool].topk(
                min(16, len(easy_pool))).indices] if len(easy_pool) else easy_pool
            negatives = torch.cat((hard, easy))
        if not len(negatives):
            continue
        pos_logits = z[anchor] @ z[positives].T / temperature
        neg_logits = z[anchor] @ z[negatives].T / temperature
        loss = torch.logsumexp(torch.cat((pos_logits, neg_logits)), 0) - \
            torch.logsumexp(pos_logits, 0)
        per_class.setdefault(int(major[anchor]), []).append(loss)
        eligible += 1
    if not per_class:
        return embeddings.sum() * 0, {"eligible_anchors": 0,
                                       "eligible_classes": 0,
                                       "mutual_neighbor_k": neighbor_count}
    return torch.stack([torch.stack(losses).mean() for losses in
                        per_class.values()]).mean(), {
        "eligible_anchors": eligible, "eligible_classes": len(per_class),
        "mutual_neighbor_k": neighbor_count}


def state_local_incidence(program_scores, patient_codes, neighbors=8):
    """Bounded cell-centered hyperedges from train-fitted NMF activities.

    Each cell may participate through its three most active programs. Ordinary
    graph edges are expression-nearest and patient-local; these hyperedges are
    program-nearest and include up to four other training patients' cells.
    A held-out patient receives only edges within its own inference graph.
    """
    scores = torch.as_tensor(program_scores, dtype=torch.float32)
    patients = torch.as_tensor(patient_codes, device=scores.device).reshape(-1)
    n = len(scores)
    if scores.ndim != 2 or n != len(patients) or n == 0 or \
            not bool(torch.isfinite(scores).all()) or bool((scores < 0).any()):
        raise ValueError("Invalid train-fitted program scores")
    width = min(3, scores.shape[1])
    chosen = scores.topk(width, dim=1).indices
    active = torch.zeros_like(scores).scatter(1, chosen, scores.gather(1, chosen))
    empty = active.sum(1) <= 0
    if bool(empty.any()):
        active[empty, chosen[empty, 0]] = 1.0
    normalized = F.normalize(active, dim=1)
    similarity = normalized @ normalized.T
    shared = (active > 0).float() @ (active > 0).float().T > 0
    same = patients[:, None] == patients[None, :]
    eye = torch.eye(n, dtype=torch.bool, device=scores.device)
    cross_scores = similarity.masked_fill(~(shared & ~same), float("-inf"))
    local_scores = similarity.masked_fill(~(shared & same & ~eye), float("-inf"))
    cross_k = min(4, neighbors, max(n - 1, 0))
    local_k = min(neighbors, max(n - 1, 0))
    incidence = torch.zeros((n, n), dtype=scores.dtype, device=scores.device)
    cols = torch.arange(n, device=scores.device)
    incidence[cols, cols] = 1.0
    if cross_k:
        values, rows = cross_scores.topk(cross_k, dim=0)
        valid = torch.isfinite(values)
        col_grid = cols[None, :].expand_as(rows)
        incidence[rows[valid], col_grid[valid]] = (0.25 + 0.75 * values[valid])
        cross_counts = valid.sum(0)
    else:
        cross_counts = torch.zeros(n, dtype=torch.long, device=scores.device)
    if local_k:
        values, rows = local_scores.topk(local_k, dim=0)
        rank = torch.arange(local_k, device=scores.device)[:, None]
        valid = torch.isfinite(values) & (rank < (neighbors - cross_counts)[None, :])
        col_grid = cols[None, :].expand_as(rows)
        incidence[rows[valid], col_grid[valid]] = (0.25 + 0.75 * values[valid])
    return incidence.unsqueeze(0)


def patient_centered_signal(tangent, patient_codes, normalizer):
    """Remove patient-wide hypergraph components before per-cell normalization."""
    codes = torch.as_tensor(patient_codes, dtype=torch.long,
                            device=tangent.device).reshape(-1)
    if len(codes) != len(tangent) or not len(codes):
        raise ValueError("Patient codes and fused cells differ")
    _, inverse = torch.unique(codes, sorted=True, return_inverse=True)
    groups = int(inverse.max()) + 1
    mean = tangent.new_zeros((groups, tangent.shape[1])).index_add(
        0, inverse, tangent)
    counts = torch.bincount(inverse, minlength=groups).clamp_min(1)
    centered = tangent - mean[inverse] / counts[inverse, None]
    return normalizer(centered)


def _shuffled_within_patient(incidence, patient_codes, global_idx):
    shuffled = incidence.clone()
    patient = torch.as_tensor(patient_codes, device=incidence.device).reshape(-1)
    for p in torch.unique(patient):
        ids = torch.nonzero(patient == p, as_tuple=False).flatten()
        if len(ids) < 2:
            continue
        origin = int(global_idx[int(ids[0])]) if global_idx is not None else int(p)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(168023 + origin)
        order = torch.randperm(len(ids), generator=generator).to(ids.device)
        shuffled[0, ids] = incidence[0, ids[order]]
    return shuffled


def patch_runtime():
    if os.environ.get("GSE_BIO_MODE") != "1":
        raise RuntimeError("GSE_BIO_MODE=1 is required")
    variant = os.environ.get("GSE_BIO_HYPER_VARIANT", "")
    cl_variant = os.environ.get("GSE_BIO_CL_VARIANT", "")
    if variant not in HYPER_VARIANTS or (cl_variant and cl_variant not in CL_VARIANTS):
        raise ValueError(f"Unknown bio intervention: {variant}, {cl_variant}")
    if cl_variant and variant == "branch_removed" and cl_variant == "cross_view":
        raise ValueError("Cross-view contrast requires a gene hypergraph")

    import hypformer
    if variant.startswith("state_local_centered"):
        def patient_centered_shared(self, x, clean_embedding):
            adapted = self.adapt_layer(clean_embedding.squeeze(0),
                                       x_manifold="hyp")
            tangent = self.manifold_hidden.logmap0(adapted)[..., 1:]
            codes = getattr(self, "_bio_current_patient_codes", None)
            if codes is None:
                raise ValueError("Patient-centered fusion requires patient codes")
            signal = patient_centered_signal(
                tangent, codes, self.cl_input_norm)
            return x + torch.sigmoid(self.cl_gate_logit) * signal, tangent

        hypformer.HypFormer._shared_input = patient_centered_shared
    if cl_variant:
        import gse_patient_modules
        gse_patient_modules.cross_patient_prototype_loss = {
            "minor_prototype": minor_prototype_loss,
            "cross_view": cross_view_loss,
            "state_mnn": state_mnn_loss,
        }[cl_variant]

    if variant == "branch_removed":
        def no_hyper_embedding(self, dataset):
            return None
        def without_hyper(self, x, clean_embedding):
            return x, torch.zeros_like(x)
        hypformer.HypFormer._clean_hyper_embedding = no_hyper_embedding
        hypformer.HypFormer._shared_input = without_hyper
        return

    archive_path = Path(os.environ["GSE_BIO_DATA_PATH"])
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    state_variant = variant.startswith("state_local")
    if state_variant:
        state_path = Path(os.environ["GSE_BIO_STATE_PATH"])
        with np.load(state_path, allow_pickle=False) as archive:
            all_weights = archive["program_scores"].astype(np.float32)
    else:
        key = "program_weights" if variant.startswith("program_") else "gene_weights"
        with np.load(archive_path, allow_pickle=False) as archive:
            all_weights = archive[key].astype(np.float32)
    shuffled = variant.endswith("_shuffled")

    def gene_hyper_embedding(self, dataset):
        x = dataset.graph["node_feat"]
        if state_variant:
            self._bio_current_patient_codes = dataset.patient_codes
        incidence = getattr(dataset, "_bio_incidence", None)
        if incidence is None or incidence.device != x.device:
            global_idx = getattr(dataset, "global_idx", None)
            rows = all_weights[global_idx] if global_idx is not None else all_weights
            if len(rows) != len(x):
                raise ValueError("Hyperedge data and graph cell order differ")
            if state_variant:
                incidence = state_local_incidence(
                    torch.as_tensor(rows, dtype=x.dtype, device=x.device),
                    dataset.patient_codes)
            else:
                incidence = torch.as_tensor(rows, dtype=x.dtype,
                                             device=x.device).unsqueeze(0)
            if shuffled:
                incidence = _shuffled_within_patient(
                    incidence, dataset.patient_codes, global_idx)
            incidence = incidence[:, :, incidence.sum(dim=1).squeeze(0) > 0]
            if (incidence.shape[2] == 0 or not torch.isfinite(incidence).all()
                    or bool((incidence.sum(dim=2) <= 0).any())):
                raise ValueError("Empty or invalid gene-program incidence")
            dataset._bio_incidence = incidence
        x_hyp = self.adv_module.manifold.expmap0(torch.cat(
            (torch.zeros_like(x[..., :1]), x), dim=-1).unsqueeze(0))
        nodes, _ = self.adv_module.get_node_edge_embeddings(
            x_hyp, incidence, project=False)
        return nodes

    hypformer.HypFormer._clean_hyper_embedding = gene_hyper_embedding
