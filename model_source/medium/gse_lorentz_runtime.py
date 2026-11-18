"""Isolated Lorentz hyperedge and contrastive interventions for GSE176078.

The input ordinary graph is built by gse_lorentz_inductive_eval. This module
changes only hyperedge incidence, hypergraph fusion, and the optional loss.
Training-patient labels are allowed only for the explicitly supervised
hierarchy prototypes; validation and test labels never enter construction.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HYPER = ("branch_removed", "encoder_only", "program_cosine",
         "program_lorentz", "hierarchy", "hierarchy_shuffled")
CONTRAST = ("off", "hierarchy")


def lorentz_inner(x, y):
    return -(x[..., :1] * y[..., :1]).sum(-1) + (x[..., 1:] * y[..., 1:]).sum(-1)


def stable_lorentz_point(point, k):
    """Recompute time in float64 to remove float32 hyperboloid drift."""
    spatial = point[..., 1:].double()
    time = torch.sqrt(k.double() + spatial.square().sum(-1, keepdim=True))
    return torch.cat((time, spatial), dim=-1)


def lorentz_cdist(x, y, k):
    """Memory-bounded Lorentz geodesic distances for points on one sheet."""
    if x.dtype != torch.float64 or y.dtype != torch.float64:
        raise ValueError("Lorentz distances require stable float64 points")
    inner = -x[:, :1] @ y[:, :1].T + x[:, 1:] @ y[:, 1:].T
    argument = (-inner / k.double()).clamp_min(1.0 + 1e-7)
    return torch.sqrt(k.double()) * torch.acosh(argument)


def normalized_lorentz_mean(points, k):
    """Extrinsic mean projected onto the future sheet, not a Frechet mean."""
    mean = points.mean(0)
    scale = torch.sqrt((-lorentz_inner(mean, mean) / k).clamp_min(1e-8))
    return mean / scale


def embed(model, x):
    manifold = model.adv_module.manifold
    tangent = torch.cat((torch.zeros_like(x[:, :1]), x), dim=-1)[None]
    mapped = manifold.expmap0(tangent)
    encoded = model.adv_module.encoder_l2(
        model.adv_module.encoder_act(model.adv_module.encoder_l1(mapped)))
    return mapped, encoded.squeeze(0)


def program_incidence(scores, patient_codes, z, manifold, metric, neighbors=8):
    """Same train-fitted candidate set and cardinality for both metrics."""
    n = len(scores)
    if n != len(patient_codes) or n != len(z) or n < 2:
        raise ValueError("Program graph data are misaligned")
    chosen = scores.topk(min(3, scores.size(1)), dim=1).indices
    active = torch.zeros_like(scores).scatter(1, chosen, scores.gather(1, chosen))
    empty = active.sum(1) <= 0
    if bool(empty.any()):
        active[empty, chosen[empty, 0]] = 1.0
    shared = (active > 0).float() @ (active > 0).float().T > 0
    patient = patient_codes.reshape(-1)
    same = patient[:, None] == patient[None, :]
    eye = torch.eye(n, dtype=torch.bool, device=scores.device)
    with torch.no_grad():
        if metric == "cosine":
            unit = F.normalize(active, dim=1)
            similarity = unit @ unit.T
        elif metric == "lorentz":
            similarity = -lorentz_cdist(z.detach(), z.detach(), manifold.k)
        else:
            raise ValueError(metric)
        incidence = scores.new_zeros((n, n))
        cols = torch.arange(n, device=scores.device)
        incidence[cols, cols] = 1.0
        cross = similarity.masked_fill(~(shared & ~same), float("-inf"))
        local = similarity.masked_fill(~(shared & same & ~eye), float("-inf"))
        cross_k = min(4, neighbors, n - 1)
        local_k = min(neighbors, n - 1)
        cross_count = torch.zeros(n, device=scores.device, dtype=torch.long)
        if cross_k:
            value, row = cross.topk(cross_k, dim=0)
            valid = torch.isfinite(value)
            column = cols[None, :].expand_as(row)
            incidence[row[valid], column[valid]] = 1.0
            cross_count = valid.sum(0)
        if local_k:
            value, row = local.topk(local_k, dim=0)
            rank = torch.arange(local_k, device=scores.device)[:, None]
            valid = torch.isfinite(value) & (rank < (neighbors - cross_count)[None, :])
            column = cols[None, :].expand_as(row)
            incidence[row[valid], column[valid]] = 1.0
    return incidence[None]


def hierarchy_prototypes(z_train, major, minor, patient, k):
    """Label-informed parent/state prototypes from training patients only."""
    parent, child, child_parent, names = [], [], [], []
    parent_labels = torch.unique(major, sorted=True)
    for parent_index, label in enumerate(parent_labels):
        mask = major == label
        parent.append(normalized_lorentz_mean(z_train[mask], k))
        state_names = sorted(set(minor[np.flatnonzero(mask.detach().cpu().numpy())]))
        for state_name in state_names:
            state = mask & torch.as_tensor(minor == state_name, device=z_train.device)
            if int(state.sum()) < 5 or len(torch.unique(patient[state])) < 2:
                continue
            child.append(normalized_lorentz_mean(z_train[state], k))
            child_parent.append(parent_index)
            names.append((int(label), str(state_name)))
    if not child:
        raise ValueError("No cross-patient fine-state prototypes")
    return (torch.stack(parent), torch.stack(child),
            torch.as_tensor(child_parent, device=z_train.device), names)


def hierarchy_incidence(z_query, parent, child, child_parent, manifold,
                        z_train=None):
    """Each cell can join broad and fine prototypes with soft radial gating."""
    with torch.no_grad():
        p, c = parent.detach(), child.detach()
        d_parent = lorentz_cdist(z_query.detach(), p, manifold.k)
        d_child = lorentz_cdist(z_query.detach(), c, manifold.k)
        reference = z_query.detach() if z_train is None else z_train.detach()
        scale = lorentz_cdist(reference, p, manifold.k).min(1).values
        tau = scale.median().clamp_min(0.1)
        n = len(z_query)
        h = z_query.new_zeros((n, len(p) + len(c)))
        n_parent = min(2, len(p))
        parent_ids = d_parent.topk(n_parent, dim=1, largest=False).indices
        rows = torch.arange(n, device=z_query.device)[:, None].expand_as(parent_ids)
        h[rows, parent_ids] = torch.exp((-d_parent[rows, parent_ids] / tau).clamp_min(-30))
        eligible = (child_parent[None, :] == parent_ids[:, :, None]).any(1)
        candidate = d_child.masked_fill(~eligible, float("inf"))
        child_count = min(2, len(c))
        distance, child_ids = candidate.topk(child_count, dim=1, largest=False)
        rows = torch.arange(n, device=z_query.device)[:, None].expand_as(child_ids)
        valid = torch.isfinite(distance)
        radius_query = manifold.dist0(z_query.detach())
        radius_child = manifold.dist0(c)
        radial = torch.sigmoid((radius_query[:, None] - radius_child[child_ids]) / tau)
        weight = torch.exp((-distance / tau).clamp(min=-30, max=0)) * radial
        h[rows[valid], len(p) + child_ids[valid]] = weight[valid]
        h = h / h.sum(1, keepdim=True).clamp_min(1e-8)
    return h[None]


def shuffled_incidence(incidence, patients, global_idx):
    """Keep each row degree and every prototype's count per patient."""
    source = incidence[0]
    result = source.clone()
    patient = patients.reshape(-1)
    degree = (source > 0).sum(1)
    for code in torch.unique(patient):
        for d in torch.unique(degree[patient == code]):
            ids = torch.nonzero((patient == code) & (degree == d),
                                as_tuple=False).flatten()
            if len(ids) < 2:
                continue
            first = int(global_idx[int(ids[0])]) if global_idx is not None else int(ids[0])
            generator = torch.Generator(device="cpu")
            generator.manual_seed(168023 + first + int(d) * 1009)
            perm = torch.randperm(len(ids), generator=generator).to(ids.device)
            result[ids] = source[ids[perm]]
    return result[None]


def hierarchical_contrast(z, major, minor, patient, manifold, temperature=0.2):
    """Leave-one-patient state positives; unrelated lineages are negatives."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Contrastive temperature must be finite and positive")
    losses, eligible = [], 0
    unique_major = torch.unique(major, sorted=True)
    major_centers = {int(y): normalized_lorentz_mean(z[major == y].detach(),
                                                     manifold.k)
                     for y in unique_major}
    for y in unique_major:
        major_mask = major == y
        negatives = torch.stack([major_centers[int(other)] for other in unique_major
                                 if int(other) != int(y)])
        for name in sorted(set(minor[np.flatnonzero(major_mask.detach().cpu().numpy())])):
            state = major_mask & torch.as_tensor(minor == name, device=z.device)
            groups = torch.unique(patient[state])
            if len(groups) < 2:
                continue
            for group in groups:
                anchors = state & (patient == group)
                others = state & (patient != group)
                positive = normalized_lorentz_mean(z[others].detach(), manifold.k)
                logits_positive = -lorentz_cdist(
                    z[anchors], positive[None], manifold.k)
                logits_negative = -lorentz_cdist(
                    z[anchors], negatives, manifold.k)
                logits = torch.cat((logits_positive, logits_negative), dim=1) / temperature
                target = torch.zeros(len(logits), dtype=torch.long, device=z.device)
                losses.append(F.cross_entropy(logits, target))
                eligible += int(anchors.sum())
    if not losses:
        return z.sum() * 0, 0
    return torch.stack(losses).mean(), eligible


def patch_runtime():
    variant = os.environ["GSE_LORENTZ_HYPER"]
    contrast = os.environ.get("GSE_LORENTZ_CL", "off")
    rank_weight = float(os.environ.get("GSE_LORENTZ_RANK_WEIGHT", "0"))
    temperature = float(os.environ.get("GSE_LORENTZ_CL_TEMPERATURE", "0.2"))
    fusion_scale = float(os.environ.get("GSE_LORENTZ_FUSION_SCALE", "1.0"))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Contrastive temperature must be finite and positive")
    if not math.isfinite(fusion_scale) or fusion_scale < 0:
        raise ValueError("Fusion scale must be finite and nonnegative")
    if variant not in HYPER or contrast not in CONTRAST or rank_weight < 0:
        raise ValueError("Unknown Lorentz intervention")
    if rank_weight and not variant.startswith("hierarchy"):
        raise ValueError("Radial ordering requires hierarchy prototypes")
    with np.load(Path(os.environ["GSE_LORENTZ_DATA"]), allow_pickle=False) as data:
        train_idx = data["train_idx"].copy()
        train_features = data["features"][train_idx].astype(np.float32)
        train_major = data["labels"][train_idx].astype(np.int64)
        train_minor = data["celltype_minor"][train_idx].astype(str)
        train_patient_names = data["patient_ids"][train_idx].astype(str)
    state_path = Path(os.environ["GSE_LORENTZ_STATE"])
    with np.load(state_path, allow_pickle=False) as state:
        all_program = state["program_scores"].astype(np.float32)
    from gse_bio_runtime import patient_centered_signal
    import gse_patient_modules
    import hypformer

    def zero_prototype_loss(embeddings, labels, patient_codes, train_idx, **kwargs):
        return embeddings.sum() * 0, {"eligible_anchors": 0,
                                       "eligible_classes": 0}
    gse_patient_modules.cross_patient_prototype_loss = zero_prototype_loss
    original_forward = hypformer.HypFormer.forward

    def clean_embedding(self, dataset):
        x = dataset.graph["node_feat"]
        x_hyp, z = embed(self, x)
        geometry = stable_lorentz_point(z, self.adv_module.manifold.k)
        self._lorentz_current_z = geometry
        self._lorentz_rank_loss = z.sum() * 0
        self._lorentz_current_patient_codes = dataset.patient_codes
        self._lorentz_last_incidence = None
        if variant == "branch_removed":
            return None
        if variant == "encoder_only":
            return z[None]
        global_idx = getattr(dataset, "global_idx", None)
        if global_idx is None:
            raise ValueError("Global cell indices required for train-fitted programs")
        if len(global_idx) != len(x):
            raise ValueError("Global indices are misaligned")
        scores = torch.as_tensor(all_program[global_idx], dtype=x.dtype, device=x.device)
        if variant.startswith("program_"):
            h = program_incidence(scores, dataset.patient_codes, geometry,
                                  self.adv_module.manifold,
                                  "cosine" if variant == "program_cosine" else "lorentz")
        else:
            if np.array_equal(global_idx, train_idx):
                z_reference = geometry
            else:
                with torch.no_grad():
                    _, reference_raw = embed(self, torch.as_tensor(
                        train_features, dtype=x.dtype, device=x.device))
                    z_reference = stable_lorentz_point(
                        reference_raw, self.adv_module.manifold.k)
            major = torch.as_tensor(train_major, dtype=torch.long, device=x.device)
            patient = torch.as_tensor(np.unique(train_patient_names,
                return_inverse=True)[1], dtype=torch.long, device=x.device)
            parent, child, child_parent, _ = hierarchy_prototypes(
                z_reference, major, train_minor, patient, self.adv_module.manifold.k)
            h = hierarchy_incidence(geometry, parent, child, child_parent,
                                    self.adv_module.manifold, z_reference)
            if variant == "hierarchy_shuffled":
                h = shuffled_incidence(h, dataset.patient_codes, global_idx)
            h = h[:, :, h.sum(dim=1).squeeze(0) > 0]
            if self.training and rank_weight:
                radius_parent = self.adv_module.manifold.dist0(parent)
                radius_child = self.adv_module.manifold.dist0(child)
                self._lorentz_rank_loss = F.relu(
                    0.1 + radius_parent[child_parent] - radius_child).mean()
            self.last_lorentz_hyper_stats = {
                "parent_count": len(parent), "state_count": len(child),
                "membership_count": int((h > 0).sum()),
                "radial_order_fraction": float((
                    self.adv_module.manifold.dist0(child) >
                    self.adv_module.manifold.dist0(parent)[child_parent]).float().mean().detach())}
        if bool((h.sum(2) <= 0).any() or (h.sum(1) <= 0).any()):
            raise ValueError("Empty hyperedge or cell")
        self._lorentz_last_incidence = h.detach()
        nodes, _ = self.adv_module.get_node_edge_embeddings(
            x_hyp, h.to(dtype=x.dtype), project=False)
        return nodes

    def shared_input(self, x, clean):
        if variant == "branch_removed":
            self._lorentz_fused_input = x.detach()
            return x, torch.zeros_like(x)
        adapted = self.adapt_layer(clean.squeeze(0), x_manifold="hyp")
        tangent = self.manifold_hidden.logmap0(adapted)[..., 1:]
        centered = patient_centered_signal(
            tangent, self._lorentz_current_patient_codes, self.cl_input_norm)
        fused = x + fusion_scale * torch.sigmoid(self.cl_gate_logit) * centered
        self._lorentz_fused_input = fused.detach()
        return fused, tangent

    def forward(self, dataset, train_idx=None):
        output = original_forward(self, dataset, train_idx=train_idx)
        if not self.training or train_idx is None:
            return output
        logits, a, b, old_loss = output
        rank_loss = rank_weight * self._lorentz_rank_loss
        cl_loss = old_loss * 0
        eligible = 0
        if contrast == "hierarchy":
            ids = torch.as_tensor(train_idx, dtype=torch.long,
                                  device=self._lorentz_current_z.device)
            major = dataset.label.squeeze(1)[ids]
            patient = dataset.patient_codes[ids]
            minor = train_minor[ids.detach().cpu().numpy()]
            cl_loss, eligible = hierarchical_contrast(
                self._lorentz_current_z[ids], major, minor, patient,
                self.adv_module.manifold, temperature=temperature)
            cl_loss = self.cl_weight * cl_loss
        self.last_lorentz_loss = {"rank": float(rank_loss.detach()),
                                  "contrast": float(cl_loss.detach()),
                                  "eligible_anchors": eligible,
                                  "cl_temperature": temperature,
                                  "fusion_scale": fusion_scale,
                                  "effective_fusion_gate": float((fusion_scale *
                                      torch.sigmoid(self.cl_gate_logit)).detach())}
        log_path = os.environ.get("GSE_LORENTZ_LOSS_LOG")
        if log_path:
            with open(log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(self.last_lorentz_loss) + "\n")
        return logits, a, b, rank_loss + cl_loss

    hypformer.HypFormer._clean_hyper_embedding = clean_embedding
    hypformer.HypFormer._shared_input = shared_input
    hypformer.HypFormer.forward = forward
