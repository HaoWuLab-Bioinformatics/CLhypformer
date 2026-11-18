"""Nonlinear patient-program hyperedges with Lorentz state transfer losses.

Fixed program identities come from genes; trainable anchors and nonlinear
edge updates are shared across patients. Patient-specific edges are inferred
from that patient's cells, including for a previously unseen patient.
"""
import torch
from torch import nn
import torch.nn.functional as F

from gse_lorentz_runtime import stable_lorentz_point, lorentz_cdist


class BiologicalStateHypergraph(nn.Module):
    def __init__(self, in_dim, programs, manifold, mode='nonlinear', width=32):
        super().__init__()
        self.mode = mode
        self.manifold = manifold
        self.width = width
        self.programs = programs
        self.input = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, width), nn.GELU())
        self.anchors = nn.Parameter(torch.randn(programs, width) * 0.05)
        self.register_buffer('expression_templates', torch.zeros(programs, in_dim))
        self.edge_update = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, width))
        self.node_update = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        self.output = nn.Linear(width, in_dim)
        self.output_norm = nn.LayerNorm(in_dim)
        self.gate = nn.Parameter(torch.tensor(-2.0))
        self.state_output = nn.Linear(width, in_dim)
        self.state_norm = nn.LayerNorm(in_dim)
        self.state_gate = nn.Parameter(torch.tensor(-2.0))
        self.last = None

    def bound(self, x):
        return x * (3.0 / x.norm(dim=-1, keepdim=True).clamp_min(3.0))

    def point(self, tangent):
        value = self.manifold.expmap0(torch.cat((torch.zeros_like(tangent[..., :1]),
                                               self.bound(tangent)), dim=-1))
        return stable_lorentz_point(value, self.manifold.k)

    def forward(self, x, prior, patients, global_idx=None):
        z = self.bound(self.input(x))
        anchor = self.bound(self.input(self.expression_templates) + self.anchors)
        # The semantic candidate mask is gene-based; soft Lorentz assignment
        # within that mask remains differentiable with respect to the anchors.
        distances = lorentz_cdist(self.point(z), self.point(anchor), self.manifold.k).to(x.dtype)
        prior = prior.to(x.dtype)
        if self.mode == 'shuffled':
            prior = prior.clone()
            degree = (prior > 0).sum(1)
            for patient in torch.unique(patients):
                for count in torch.unique(degree[patients == patient]):
                    ids = torch.where((patients == patient) & (degree == count))[0]
                    generator = torch.Generator(device='cpu').manual_seed(176078 + int(ids[0]))
                    perm = torch.randperm(len(ids), generator=generator).to(ids.device)
                    prior[ids] = prior[ids[perm]].clone()
        logits = prior.clamp_min(1e-8).log() - distances / 0.5
        logits = logits.masked_fill(prior <= 0, -1e4)
        h = logits.softmax(1) * (prior > 0)
        h = h / h.sum(1, keepdim=True).clamp_min(1e-8)
        updated = torch.empty_like(z)
        state_message = torch.zeros_like(z)
        edges, support = [], []
        for patient in torch.unique(patients, sorted=True):
            ids = torch.where(patients == patient)[0]
            local, membership = z[ids], h[ids]
            mass = membership.sum(0)
            edge = anchor
            for _ in range(2):
                aggregate = (membership.T @ local) / mass[:, None].clamp_min(1e-6)
                if self.mode in ('nonlinear', 'shuffled'):
                    edge = self.bound(edge + 0.25 * torch.tanh(self.edge_update(
                        torch.cat((aggregate, edge, anchor), dim=-1))))
                    message = membership @ edge
                elif self.mode == 'linear':
                    # Given H, this message is an effective weighted graph
                    # propagation. It has no nonlinear edge-state update.
                    edge = aggregate
                    message = membership @ edge
                else:
                    edge = anchor * 0
                    message = torch.zeros_like(local)
                delta = 0.25 * torch.tanh(self.node_update(torch.cat((local, message), dim=-1)))
                # With zero membership the local update equals the encoder
                # control, rather than changing its depth for inactive cells.
                local = self.bound(local + delta)
            updated[ids] = local
            if self.mode in ('nonlinear', 'shuffled', 'linear'):
                state_message[ids] = membership @ edge
            edges.append(edge)
            support.append((membership > 0).sum(0))
        edge_tangent = torch.stack(edges)
        correction = self.output_norm(self.output(updated))
        fused = x + torch.sigmoid(self.gate) * correction
        if self.mode in ('nonlinear', 'shuffled', 'linear'):
            # A separate state channel keeps the program message from being
            # overwhelmed by the larger local encoder representation.
            semantic = self.state_norm(self.state_output(state_message + updated - z))
            semantic = semantic * (h.sum(1, keepdim=True) > 0)
            fused = fused + torch.sigmoid(self.state_gate) * semantic
        if self.mode == 'removed':
            fused = x
        self.last = dict(cells=self.point(updated + state_message), edges=self.point(edge_tangent),
                         edge_tangent=edge_tangent, membership=h, prior=prior,
                         support=torch.stack(support), fused=fused,
                         original=z)
        return fused

    def contrast(self, positive, negative_program, patients, temperature=0.5):
        last = self.last
        edges = last['edges'].flatten(0, 1)
        support = last['support'].flatten() >= 3
        m = self.programs
        program_ids = torch.arange(len(edges), device=edges.device) % m
        valid = support[:, None] & support[None, :]
        pos = positive & valid
        neg = negative_program[program_ids[:, None], program_ids[None, :]] & valid
        logits = -lorentz_cdist(edges, edges, self.manifold.k) / temperature
        rows = pos.any(1) & neg.any(1)
        per_edge = (torch.logsumexp(logits.masked_fill(~(pos | neg), -1e4), 1) -
                    torch.logsumexp(logits.masked_fill(~pos, -1e4), 1))
        balanced = [per_edge[rows & (program_ids == program)].mean()
                    for program in range(m) if bool((rows & (program_ids == program)).any())]
        zero = edges.sum() * 0
        edge_loss = torch.stack(balanced).mean() if balanced else zero
        cell_losses, eligible_cells = [], 0
        for group in torch.unique(patients, sorted=True):
            mask = patients == group
            matches = pos[int(group) * m:(int(group) + 1) * m].to(last['edge_tangent'].dtype)
            denominator = matches.sum(1)
            reference = matches @ last['edge_tangent'].flatten(0, 1) / denominator[:, None].clamp_min(1)
            target = last['prior'][mask].detach() * (denominator > 0)[None]
            target = target / target.sum(1, keepdim=True).clamp_min(1e-8)
            active = target > 0
            unrelated = active.float() @ (~negative_program).float() == 0
            allowed = ((active | unrelated) & (denominator > 0)[None])
            query_logits = -lorentz_cdist(last['cells'][mask], self.point(reference),
                                         self.manifold.k) / temperature
            query_logits = query_logits.masked_fill(~allowed, -1e4)
            eligible = (target.sum(1) > 0) & (allowed & ~active).any(1)
            loss = -(target * F.log_softmax(query_logits, 1)).sum(1)
            if bool(eligible.any()):
                cell_losses.append(loss[eligible].mean())
                eligible_cells += int(eligible.sum())
        cell_loss = torch.stack(cell_losses).mean() if cell_losses else zero
        diagnostics = dict(edge_loss=float(edge_loss.detach()), cell_loss=float(cell_loss.detach()),
                           eligible_edges=int(rows.sum()), eligible_cells=eligible_cells,
                           matched_directed_pairs=int(pos.sum()))
        return edge_loss + 0.5 * cell_loss, diagnostics
