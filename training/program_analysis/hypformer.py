import pdb
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from manifolds.hyp_layer import HypLinear, HypLayerNorm, HypActivation, HypDropout, HypCLS
from manifolds.lorentz import Lorentz
from geoopt import ManifoldParameter
from gnns import GraphConv, GCN


class AdaptiveCurvatureLayer(nn.Module):
    """
    自适应曲率模块：将固定曲率改为可学习参数。
    """

    def __init__(self, initial_k=1.0):
        super().__init__()
        # 初始化为 nn.Parameter，以便优化器能够识别并更新
        self.k_param = nn.Parameter(torch.tensor([float(initial_k)]))

    def forward(self):
        k_val = F.softplus(self.k_param) + 1e-5
        # 限制曲率在一个安全的双曲区间内，确保流形结构的数学稳定性
        return torch.clamp(k_val, min=0.4, max=3.0)


class HypCNHCL_Loss(nn.Module):
    def __init__(self, manifold, tau=0.5, w_e=1.0, w_m=1.0, tau_plus=0.1, tau_hard=0.2, cl_space='hyp'):
        super().__init__()
        self.manifold = manifold
        self.cl_space = cl_space

        # 1. 将 tau 和 tau_hard 修改为可学习的参数 (nn.Parameter)
        self.tau = nn.Parameter(torch.tensor([float(tau)]))
        self.tau_hard = nn.Parameter(torch.tensor([float(tau_hard)]))

        self.w_e = w_e
        self.w_m = w_m
        # tau_plus 代表類別的先驗概率 (即隨機抽到同類的機率)
        self.tau_plus = tau_plus

    def info_nce_loss(self, z1, z2, topology_mask=None):
        """基於洛倫茲內積優化的去偏 (Debiased) InfoNCE，並引入硬負樣本挖掘與拓扑感知"""
        # 确保温度系数始终为正
        current_tau = F.softplus(self.tau) + 1e-4
        current_tau_hard = F.softplus(self.tau_hard) + 1e-4

        z1 = z1.reshape(-1, z1.size(-1))
        z2 = z2.reshape(-1, z2.size(-1))
        num_items = z1.size(0)

        # 处理 topology_mask 的形状，兼容 (T, N, N) -> (N, N)
        if topology_mask is not None:
            if topology_mask.dim() == 3 and topology_mask.size(0) == 1:
                topology_mask = topology_mask.squeeze(0)

        # 採樣優化，處理大規模節點
        if num_items > 5000:
            sample_idx = torch.randperm(num_items, device=z1.device)[:3000]
            z1 = z1[sample_idx]
            z2 = z2[sample_idx]
            num_items = z1.size(0)
            # 如果发生了采样，拓扑矩阵也必须同步采样对齐
            if topology_mask is not None:
                topology_mask = topology_mask[sample_idx][:, sample_idx]

        if self.cl_space == 'hyp':
            z1_time, z1_space = z1[:, 0:1], z1[:, 1:]
            z2_time, z2_space = z2[:, 0:1], z2[:, 1:]

            inner = torch.matmul(z1_space, z2_space.t()) - torch.matmul(z1_time, z2_time.t())
            inner_prod = -self.manifold.k * inner
            inner_prod_clipped = torch.clamp(inner_prod, min=1.0001)
            dist_sq = (1.0 / self.manifold.k) * (torch.acosh(inner_prod_clipped) ** 2)

            sim = -dist_sq / current_tau
        elif self.cl_space == 'euc':
            z1_euc = self.manifold.logmap0(z1)[..., 1:]
            z2_euc = self.manifold.logmap0(z2)[..., 1:]

            z1_norm = F.normalize(z1_euc, p=2, dim=1)
            z2_norm = F.normalize(z2_euc, p=2, dim=1)

            sim = torch.matmul(z1_norm, z2_norm.t()) / current_tau
            dist_sq = torch.cdist(z1_euc, z2_euc, p=2.0) ** 2
        else:
            raise ValueError("cl_space must be 'hyp' or 'euc'")

        sim = torch.clamp(sim, min=-50.0, max=0.0 if self.cl_space == 'hyp' else 50.0)

        # 3. 數值穩定性優化 (Log-Sum-Exp Trick)
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim_shifted = sim - sim_max.detach()

        # 4. 提取正樣本相似度
        pos_sim = torch.diag(sim_shifted)
        exp_pos = torch.exp(pos_sim)

        # ================== 核心修改：双曲硬负样本 + 拓扑感知掩码 ==================
        # 距離越小（越 Hard），權重越大
        w_neg = torch.exp(-dist_sq / current_tau_hard)

        # 构造初始掩码
        mask = torch.ones_like(w_neg)

        # 引入拓扑感知逻辑：如果在关联矩阵 H 中相连 (权重 > 0)，则在 mask 中置为 0
        if topology_mask is not None:
            if topology_mask.shape == mask.shape:
                mask[topology_mask > 0] = 0.0

        # 排除正样本自身（对角线）
        mask.fill_diagonal_(0.0)

        # 計算加權後的負樣本指數和
        exp_sim = torch.exp(sim_shifted)
        weighted_neg_exp = exp_sim * w_neg * mask
        neg_sum = weighted_neg_exp.sum(dim=1)
        # ======================================================================

        # 5. 去偏估計與理論下界保護
        N = num_items
        debiased_neg_sum = (neg_sum - N * self.tau_plus * exp_pos) / (1.0 - self.tau_plus)

        # 【关键修复】：将 math.exp 替换为 torch.exp，以支持可学习的 Tensor 梯度传播
        bound = N * torch.exp(-1.0 / current_tau) * torch.exp(-sim_max.squeeze())
        debiased_neg_sum = torch.clamp(debiased_neg_sum, min=1e-7)
        debiased_neg_sum = torch.max(debiased_neg_sum, bound)

        # 6. 計算最終 Loss
        loss = torch.log(exp_pos + debiased_neg_sum + 1e-8) - pos_sim

        return loss.mean()

    def forward(self, zn1, zn2, ze1, ze2, H=None):
        """計算節點、超邊及交叉視圖的對比損失，接收拓扑关联矩阵 H"""
        # 将拓扑矩阵 H 传入节点级别的对比损失中过滤 False Negatives
        l_n = (self.info_nce_loss(zn1, zn2, H) + self.info_nce_loss(zn2, zn1, H)) / 2
        l_e = (self.info_nce_loss(ze1, ze2) + self.info_nce_loss(ze2, ze1)) / 2
        l_m = (self.info_nce_loss(zn1, ze2) + self.info_nce_loss(zn2, ze1)) / 2
        return l_n + self.w_e * l_e + self.w_m * l_m

    def supervised_node_loss(self, z1, z2, labels, train_idx,
                             group_ids=None, cross_group=False,
                             coarse_labels=None, mask_same_major=False,
                             pair_policy='diag_fixed', patient_balance=False):
        """Contrast only training nodes; same-class nodes are all positives."""
        if labels is None or train_idx is None:
            raise ValueError("Supervised contrastive loss requires training labels and indices")
        z1 = z1.reshape(-1, z1.size(-1))
        z2 = z2.reshape(-1, z2.size(-1))
        train_idx = train_idx.to(device=z1.device, dtype=torch.long).reshape(-1)
        if train_idx.numel() == 0:
            raise ValueError("Supervised contrastive loss requires nonempty train_idx")
        y = labels.reshape(-1).to(z1.device)[train_idx]
        z1, z2 = z1[train_idx], z2[train_idx]
        tau = F.softplus(self.tau) + 1e-4
        if self.cl_space == 'hyp':
            inner = z1[:, 1:] @ z2[:, 1:].T - z1[:, :1] @ z2[:, :1].T
            cosh_distance = (-inner / self.manifold.k).clamp(min=1.000001)
            distance = torch.acosh(cosh_distance) * torch.sqrt(self.manifold.k)
            similarity = -distance.square() / tau
        else:
            z1 = F.normalize(self.manifold.logmap0(z1)[:, 1:], dim=-1)
            z2 = F.normalize(self.manifold.logmap0(z2)[:, 1:], dim=-1)
            similarity = (z1 @ z2.T) / tau
        same_type = y[:, None].eq(y[None, :])
        diagonal = torch.eye(y.numel(), dtype=torch.bool, device=y.device)
        if cross_group:
            if group_ids is None:
                raise ValueError("Cross-patient contrast requires patient IDs")
            groups = group_ids.reshape(-1).to(z1.device)[train_idx]
            same_patient = groups[:, None].eq(groups[None, :])
            positives = same_type & ~same_patient
            if pair_policy == 'legacy':
                missing = ~positives.any(dim=1)
                if missing.any():
                    rows = missing.nonzero(as_tuple=True)[0]
                    positives[rows, rows] = True
            elif pair_policy == 'diag_fixed':
                positives = positives | diagonal
            else:
                raise ValueError(f'Unknown contrastive pair policy: {pair_policy}')
            ignore = same_type & same_patient
            ignore.fill_diagonal_(False)
        else:
            positives = same_type
            ignore = torch.zeros_like(positives)
        if mask_same_major:
            if coarse_labels is None:
                raise ValueError('Coarse labels required for hierarchical negative masking')
            coarse = coarse_labels.reshape(-1).to(y.device)[train_idx]
            ignore = ignore | (coarse[:, None].eq(coarse[None, :]) & ~same_type)
        if (ignore & positives).any():
            raise ValueError('A contrastive pair cannot be both positive and ignored')
        similarity = similarity.masked_fill(ignore, float('-inf'))
        positive_weights = positives.to(similarity.dtype)
        if patient_balance:
            if not cross_group:
                raise ValueError('Patient-balanced positives require cross-patient mode')
            _, patient_index = torch.unique(groups, return_inverse=True)
            patient_one_hot = F.one_hot(patient_index).to(similarity.dtype)
            forward_counts = positive_weights @ patient_one_hot
            reverse_weights = positive_weights.T
            reverse_counts = reverse_weights @ patient_one_hot
            reverse_weights = reverse_weights / reverse_counts[:, patient_index].clamp(min=1)
            positive_weights = positive_weights / forward_counts[:, patient_index].clamp(min=1)
        else:
            reverse_weights = positive_weights.T
        positive_counts = positive_weights.sum(dim=1).clamp(min=1)
        log_prob_12 = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
        log_prob_21 = similarity.T - torch.logsumexp(similarity.T, dim=1, keepdim=True)
        loss_12 = -(log_prob_12.masked_fill(positive_weights == 0, 0.0)
                    * positive_weights).sum(dim=1) / positive_counts
        loss_21 = -(log_prob_21.masked_fill(reverse_weights == 0, 0.0)
                    * reverse_weights).sum(dim=1) / reverse_weights.sum(dim=1).clamp(min=1)
        return (loss_12.mean() + loss_21.mean()) / 2


class HypHypergraphAdversarialModule(nn.Module):
    def __init__(self, manifold, in_channels, hidden_channels, k=5, drop_rate=0.2, tau=0.5,
                 knn_metric='hyp', drop_strategy='critical', cl_space='hyp',
                 graph_build_method='knn', eps_threshold=1.5,
                 walk_length=10, num_walks=5, capture_topology=False,
                 graph_source='input', feature_noise_scale=0.1):
        super().__init__()
        self.manifold = manifold
        self.k_neighbors = k
        self.base_drop_rate = drop_rate  # 设为基准 drop_rate
        self.knn_metric = knn_metric
        self.drop_strategy = drop_strategy
        self.cl_space = cl_space
        self.graph_build_method = graph_build_method
        self.eps_threshold = eps_threshold
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.cl_tau = tau
        self.capture_topology = capture_topology
        self.last_snapshot = None
        self.graph_source = graph_source
        self.feature_noise_scale = feature_noise_scale
        self.freeze_eval_graph = False
        self.cached_eval_h = None
        self.encoder_l1 = HypLinear(self.manifold, in_channels, hidden_channels)
        self.encoder_act = HypActivation(self.manifold, activation=F.relu)
        self.encoder_l2 = HypLinear(self.manifold, hidden_channels, hidden_channels)

        self.proj_node_l1 = HypLinear(self.manifold, hidden_channels, hidden_channels)
        self.proj_node_act = HypActivation(self.manifold, activation=F.relu)
        self.proj_node_l2 = HypLinear(self.manifold, hidden_channels, hidden_channels)

        self.proj_edge_l1 = HypLinear(self.manifold, hidden_channels, hidden_channels)
        self.proj_edge_act = HypActivation(self.manifold, activation=F.relu)
        self.proj_edge_l2 = HypLinear(self.manifold, hidden_channels, hidden_channels)
        self.proj_norm = HypLayerNorm(self.manifold, hidden_channels)
        self.contrast_loss = HypCNHCL_Loss(manifold=self.manifold, tau=tau, cl_space=self.cl_space)

        # 判别器在欧式空间
        '''
        self.node_discriminator = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )
        self.edge_discriminator = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )
        '''
        # 判别器在双曲空间
        self.node_discriminator = nn.Sequential(
            HypLinear(self.manifold, hidden_channels, hidden_channels // 2),
            HypActivation(self.manifold, activation=F.relu),
            HypLinear(self.manifold, hidden_channels // 2, 1)  # 输出 1 维的双曲特征
        )
        self.edge_discriminator = nn.Sequential(
            HypLinear(self.manifold, hidden_channels, hidden_channels // 2),
            HypActivation(self.manifold, activation=F.relu),
            HypLinear(self.manifold, hidden_channels // 2, 1)
        )
    @torch.no_grad()
    def build_knn_undirected_graph_hyp(self, x_hyp):
        T, N, D = x_hyp.shape
        x_time = x_hyp[..., 0:1]
        x_space = x_hyp[..., 1:]

        inner = torch.matmul(x_space, x_space.transpose(1, 2)) - torch.matmul(x_time, x_time.transpose(1, 2))
        inner_prod = -inner / self.manifold.k
        inner_prod_clipped = torch.clamp(inner_prod, min=1.0001)
        dist = torch.acosh(inner_prod_clipped) * torch.sqrt(self.manifold.k)

        _, indices = torch.topk(dist, k=self.k_neighbors + 1, largest=False)
        H = torch.zeros(T, N, N, device=x_hyp.device)
        H.scatter_(2, indices, 1.0)
        H = torch.max(H, H.transpose(1, 2))
        return H

    @torch.no_grad()
    def build_knn_undirected_graph_euc(self, x):
        """【新增】：欧式空间 KNN 建图作为消融 Baseline"""
        T, N, D = x.shape

        # 计算欧式距离矩阵
        dist = torch.cdist(x, x, p=2.0)

        # 选取最近的 K+1 个邻居 (包含自身)
        _, indices = torch.topk(dist, k=self.k_neighbors + 1, largest=False)
        H = torch.zeros(T, N, N, device=x.device)
        H.scatter_(2, indices, 1.0)

        # 保证无向图的对称性
        H = torch.max(H, H.transpose(1, 2))
        return H

    @torch.no_grad()
    def build_eps_undirected_graph_hyp(self, x_hyp):
        """双曲空间基于距离阈值(Epsilon)的建图"""
        T, N, D = x_hyp.shape
        x_time = x_hyp[..., 0:1]
        x_space = x_hyp[..., 1:]

        inner = torch.matmul(x_space, x_space.transpose(1, 2)) - torch.matmul(x_time, x_time.transpose(1, 2))
        inner_prod = -inner / self.manifold.k
        inner_prod_clipped = torch.clamp(inner_prod, min=1.0001)
        dist = torch.acosh(inner_prod_clipped) * torch.sqrt(self.manifold.k)

        H = (dist < self.eps_threshold).to(x_hyp.dtype)
        H = torch.max(H, H.transpose(1, 2))
        return H

    @torch.no_grad()
    def build_knn_undirected_graph_euc(self, x):
        """
        在欧式空间中构建无向 KNN 图（消融实验 Baseline）
        支持输入维度: (T, N, D) 或 (N, D)
        """
        # 1. 自动兼容二维输入，保持与 forward 逻辑中的三维张量一致性
        if x.dim() == 2:
            x = x.unsqueeze(0)

        T, N, D = x.shape

        # 2. 计算欧氏距离矩阵 (T, N, N)
        # torch.cdist 在 PyTorch 中经过了高度优化，计算大规模节点距离时比手动拓展维度快得多
        dist = torch.cdist(x, x, p=2.0)

        # 3. 选取最近的 K + 1 个邻居 (包含节点自身)
        # largest=False 确保选出的是距离最小的邻居
        _, indices = torch.topk(dist, k=self.k_neighbors + 1, largest=False)

        # 4. 构造初始的邻接/关联矩阵 H (T, N, N)
        H = torch.zeros(T, N, N, device=x.device)
        H.scatter_(2, indices, 1.0)

        # 5. 保证无向图的对称性（与双曲建图逻辑完全对齐）
        H = torch.max(H, H.transpose(1, 2))

        return H

    @torch.no_grad()
    def build_random_walk_undirected_graph(self, x):
        """
        经典的随机游走建图（不区分双曲或欧式几何特性，纯粹基于输入特征的马尔可夫转移）
        支持输入维度: (T, N, D) 或 (N, D)
        """
        # 1. 自动兼容二维输入，保持张量维度一致性
        if x.dim() == 2:
            x = x.unsqueeze(0)

        T, N, D = x.shape

        # 2. 计算特征空间中的标准欧氏距离 (经典 RBF 核度量)
        # 无论传入的是 x 还是 x_hyp，这里都直接将其作为普通数值张量处理
        dist = torch.cdist(x, x, p=2.0)

        # 3. 转换为相似度 (转移权重)
        sim = torch.exp(-dist)

        # 消除自环，避免随机游走时原地踏步
        # 使用高级索引替换 sim.fill_diagonal_(0.0)，确保在 3D Tensor 下精确命中对角线
        sim[:, torch.arange(N), torch.arange(N)] = 0.0

        # 4. 行归一化，得到标准的马尔可夫转移概率矩阵 (每行和为1)
        trans_prob = sim / (sim.sum(dim=2, keepdim=True) + 1e-8)

        # 5. 初始化状态扩散矩阵 (单位阵，代表所有游走均从节点自身出发)
        H_diffuse = torch.eye(N, device=x.device).unsqueeze(0).expand(T, N, N).clone()

        # 6. 执行随机游走扩散 (矩阵连乘模拟游走步数)
        for _ in range(self.walk_length):
            H_diffuse = torch.bmm(H_diffuse, trans_prob)

        # 7. 阈值截断，并保证无向图的对称性
        H = (H_diffuse > 0.2).to(x.dtype)
        H = torch.max(H, H.transpose(1, 2))

        return H

    def get_node_edge_embeddings(self, x_hyp, H, project=True):
        z_base = self.encoder_l2(self.encoder_act(self.encoder_l1(x_hyp)))
        z_tangent = self.manifold.logmap0(z_base)

        d_v = H.sum(dim=2).clamp(min=1e-5)
        d_e = H.sum(dim=1).clamp(min=1e-5)
        D_v_invsqrt = torch.diag_embed(1.0 / torch.sqrt(d_v))
        D_e_inv = torch.diag_embed(1.0 / d_e)

        # Symmetric degree normalization avoids degree-dependent amplification.
        scaled_node = torch.matmul(D_v_invsqrt, z_tangent)
        ze_tangent = torch.matmul(D_e_inv, torch.matmul(H.transpose(1, 2), scaled_node))
        zn_tangent = torch.matmul(D_v_invsqrt, torch.matmul(H, ze_tangent))

        ze_hyp = self.manifold.expmap0(ze_tangent)
        zn_hyp = self.manifold.expmap0(zn_tangent)

        if not project:
            return zn_hyp, ze_hyp

        proj_zn = self.proj_node_l2(self.proj_norm(self.proj_node_act(self.proj_node_l1(zn_hyp))))
        proj_ze = self.proj_edge_l2(self.proj_norm(self.proj_edge_act(self.proj_edge_l1(ze_hyp))))
        return proj_zn, proj_ze

    def adversarial_drop_incidence(self, H, node_scores, edge_scores):
        if self.base_drop_rate == 0.0:
            return H

        critical_weight = (node_scores + edge_scores.transpose(1, 2)) / 2.0

        # 逻辑反转：分数越高（越关键），丢弃概率越大
        # 注意限制最大丢弃概率，避免图结构完全崩溃
        max_drop_prob = 0.8
        dynamic_drop_prob = torch.clamp(critical_weight * self.base_drop_rate * 2.0, max=max_drop_prob)

        mask = (torch.rand_like(H) > dynamic_drop_prob).to(H.dtype)
        return H * mask

    def hyperbolic_feature_noise(self, x_hyp, noise_scale=0.1):
        # 1. 映射到切空间 (原点)
        tangent_x = self.manifold.logmap0(x_hyp)

        # 2. 仅在空间维度生成高斯噪声，时间维度保持为0 (Lorentz流形要求)
        noise_space = torch.randn_like(tangent_x[..., 1:]) * noise_scale
        noise_time = torch.zeros_like(tangent_x[..., 0:1])
        noise = torch.cat([noise_time, noise_space], dim=-1)

        # 3. 切空间加噪并映射回双曲空间
        perturbed_x_hyp = self.manifold.expmap0(tangent_x + noise)
        return perturbed_x_hyp
    #判别器在欧氏空间
    '''
    def get_critical_scores(self, x_hyp, H):
        # 1. 获取双曲层输出
        z_base = self.encoder_l2(self.encoder_act(self.encoder_l1(x_hyp)))

        # 2. 映射到切空间 (此时维度是 hidden_channels + 1)
        z_tangent_full = self.manifold.logmap0(z_base)

        # 【关键修复】：只取 1 维之后的部分，使其维度重新回到 hidden_channels (256)
        # Lorentz 模型的 logmap0 后的第 0 维通常是时间轴常数或与空间轴冗余的信息
        z_tangent = z_tangent_full[..., 1:]

        d_v = H.sum(dim=2).clamp(min=1e-5)
        d_e = H.sum(dim=1).clamp(min=1e-5)
        D_v_invsqrt = torch.diag_embed(1.0 / torch.sqrt(d_v))
        D_e_inv = torch.diag_embed(1.0 / d_e)

        # 3. 在切空间进行聚合
        ze_tangent = torch.matmul(D_e_inv, torch.matmul(H.transpose(1, 2), z_tangent))
        zn_tangent = torch.matmul(D_v_invsqrt, torch.matmul(H, ze_tangent))

        # 4. 现在维度是 (T, N, 256)，可以顺利通过 Discriminator
        node_scores = self.node_discriminator(zn_tangent)
        edge_scores = self.edge_discriminator(ze_tangent)

        return node_scores, edge_scores
    '''
    # 判别器在双曲空间
    def get_critical_scores(self, x_hyp, H):
        # 1. 获取双曲表示 (zn_hyp, ze_hyp)
        # 注意：这里直接使用你 get_node_edge_embeddings 逻辑中的中间变量
        # 为了演示，我简化了逻辑流程：
        z_base = self.encoder_l2(self.encoder_act(self.encoder_l1(x_hyp)))
        z_tangent = self.manifold.logmap0(z_base)

        d_v = H.sum(dim=2).clamp(min=1e-5)
        d_e = H.sum(dim=1).clamp(min=1e-5)
        D_v_invsqrt = torch.diag_embed(1.0 / torch.sqrt(d_v))
        D_e_inv = torch.diag_embed(1.0 / d_e)

        ze_tangent = torch.matmul(D_e_inv, torch.matmul(H.transpose(1, 2), z_tangent))
        zn_tangent = torch.matmul(D_v_invsqrt, torch.matmul(H, ze_tangent))

        # 映射回双曲流形
        ze_hyp = self.manifold.expmap0(ze_tangent)
        zn_hyp = self.manifold.expmap0(zn_tangent)

        # 2. 使用双曲判别器进行处理
        # 此时输出仍然在流形上 (维度为 1+1，即 Lorentz 模型的 2 维)
        node_feat_hyp = self.node_discriminator(zn_hyp)
        edge_feat_hyp = self.edge_discriminator(ze_hyp)

        # 3. 映射回切空间并取空间维度，得到标量分数
        # 在 Lorentz 模型中，1维特征经过 logmap 后，索引 [..., 1:] 得到的就是最后的评分标量
        node_scores = torch.sigmoid(self.manifold.logmap0(node_feat_hyp)[..., 1:])
        edge_scores = torch.sigmoid(self.manifold.logmap0(edge_feat_hyp)[..., 1:])

        return node_scores, edge_scores
    # --- 修改：具有关键感知的关联矩阵扰动函数 ---
    def critical_aware_drop_incidence(self, H, node_scores, edge_scores):
        if self.base_drop_rate == 0.0:
            return H

        # 结合节点分数和超边分数，构建 (T, N, N) 的权重矩阵
        # 节点 v 与 超边 e 的连接关键度等于两者分数的均值
        critical_weight = (node_scores + edge_scores.transpose(1, 2)) / 2.0

        # 动态调整 Drop 概率：
        # 如果分数趋近于 1 (极其关键)，动态 Drop 概率趋近于 0，必定保留
        # 如果分数趋近于 0 (不关键)，动态 Drop 概率达到最大的 base_drop_rate
        dynamic_drop_prob = self.base_drop_rate * (1.0 - critical_weight)

        # 生成掩码：随机数 > 动态 Drop 概率的元素保留为 1，否则为 0
        mask = (torch.rand_like(H) > dynamic_drop_prob).to(H.dtype)

        return H * mask

    def random_drop_incidence(self, H):
        """最基础的随机丢弃 (作为消融 Baseline)"""
        if self.base_drop_rate == 0.0:
            return H
        mask = (torch.rand_like(H) > self.base_drop_rate).to(H.dtype)
        diagonal = torch.arange(H.size(-1), device=H.device)
        mask[:, diagonal, diagonal] = 1.0
        return H * mask
    def hard_drop_incidence(self, H, node_scores, edge_scores):
        """
        困难视图扰动：倾向于丢弃关键边（高 score 对应高 Drop 概率）。
        """
        if self.base_drop_rate == 0.0:
            return H

        # 结合节点和超边的分数
        critical_weight = (node_scores + edge_scores.transpose(1, 2)) / 2.0

        # 逻辑反转：分数越高，丢弃概率越大
        # 设置最大丢弃上限（例如 0.7），确保不会将图完全置空
        max_drop_prob = 0.7
        # 分数接近 1 时，丢弃概率趋近 max_drop_prob；分数接近 0 时，丢弃概率趋近 0
        dynamic_drop_prob = critical_weight * max_drop_prob

        mask = (torch.rand_like(H) > dynamic_drop_prob).to(H.dtype)
        return H * mask

    def get_embedding_affinity(self, x_hyp, differentiable=False):
        """Pairwise similarity of learned node embeddings in [0, 1]."""
        if differentiable:
            z_base = self.encoder_l2(self.encoder_act(self.encoder_l1(x_hyp)))
        else:
            with torch.no_grad():
                z_base = self.encoder_l2(self.encoder_act(self.encoder_l1(x_hyp)))
        z_tangent = self.manifold.logmap0(z_base)[..., 1:]
        z_norm = z_tangent.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        z_unit = z_tangent / z_norm
        sim = torch.matmul(z_unit, z_unit.transpose(1, 2))
        return (sim + 1.0) / 2.0

    def embedding_homophily_loss(self, x, labels, train_idx):
        """Supervise embedding similarity with labeled same/different pairs."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x_hyp = self.manifold.expmap0(
            torch.cat([torch.zeros(x.shape[:-1] + (1,), device=x.device), x], dim=-1))
        affinity = self.get_embedding_affinity(x_hyp, differentiable=True)
        if affinity.dim() == 3 and affinity.size(0) == 1:
            affinity = affinity[0]

        if labels.dim() == 2 and labels.size(1) > 1:
            labels = labels.argmax(dim=1)
        labels = labels.reshape(-1)
        train_mask = torch.zeros(x.size(1), device=x.device, dtype=torch.bool)
        train_mask[train_idx.to(x.device)] = True
        same = ((labels[:, None] == labels[None, :])
                & train_mask[:, None] & train_mask[None, :])
        diff = ((labels[:, None] != labels[None, :])
                & train_mask[:, None] & train_mask[None, :])
        same.fill_diagonal_(False)
        diff.fill_diagonal_(False)

        pos = affinity[same].mean() if same.any() else affinity.new_tensor(0.0)
        neg = affinity[diff].mean() if diff.any() else affinity.new_tensor(0.0)
        return neg - pos

    def topk_drop_incidence(self, H, affinity):
        """Keep the top-k most critical entries per hyperedge column."""
        if self.base_drop_rate == 0.0:
            return H

        keep_ratio = 1.0 - self.base_drop_rate
        weights = affinity.masked_fill(H == 0, float('-inf'))
        edge_sizes = H.sum(dim=2)
        max_k = int(edge_sizes.max().clamp(min=1).item())
        k_per_column = (edge_sizes * keep_ratio).ceil().to(torch.long).clamp(min=1)
        _, indices = torch.topk(weights, k=max_k, dim=2)
        rank = torch.arange(max_k, device=H.device).reshape(1, 1, max_k)
        keep = rank < k_per_column.unsqueeze(-1)
        mask = torch.zeros_like(H)
        mask.scatter_(2, indices, keep.to(H.dtype))
        H1 = H * mask
        return torch.max(H1, H1.transpose(1, 2))

    def forward(self, x, labels=None, train_idx=None, objective='legacy',
                return_clean=False, group_ids=None, cross_group=False,
                coarse_labels=None, mask_same_major=False,
                pair_policy='diag_fixed', patient_balance=False,
                incidence_override=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # 将输入映射到双曲空间
        x_hyp = self.manifold.expmap0(torch.cat([torch.zeros(x.shape[:-1] + (1,), device=x.device), x], dim=-1))

        if incidence_override is not None:
            if (incidence_override.ndim != 3 or
                    incidence_override.shape != (1, x.size(1), x.size(1)) or
                    incidence_override.device != x.device):
                raise ValueError('Patient incidence must align with node features')
            H_orig = incidence_override
        elif self.graph_build_method == 'knn':
            if self.knn_metric == 'hyp':
                H_orig = self.build_knn_undirected_graph_hyp(x_hyp)
            elif self.knn_metric == 'euc':
                H_orig = self.build_knn_undirected_graph_euc(x)
            else:
                raise ValueError(f"不支持的 KNN 度量方式: {self.knn_metric}")
        elif self.graph_build_method == 'random_walk':
            H_orig = self.build_random_walk_undirected_graph(x)
        else:
            raise ValueError(f"不支持的建图方式: {self.graph_build_method}")

        H_input = H_orig.detach().clone()
        if incidence_override is None and self.graph_source == 'learned':
            z_base = self.encoder_l2(self.encoder_act(self.encoder_l1(x_hyp)))
            if self.graph_build_method == 'knn':
                if self.knn_metric == 'hyp':
                    H_orig = self.build_knn_undirected_graph_hyp(z_base)
                else:
                    z_euc = self.manifold.logmap0(z_base)[..., 1:]
                    H_orig = self.build_knn_undirected_graph_euc(z_euc)
            else:
                z_euc = self.manifold.logmap0(z_base)[..., 1:]
                H_orig = self.build_random_walk_undirected_graph(z_euc)

        if not self.training and self.freeze_eval_graph:
            if self.cached_eval_h is None:
                self.cached_eval_h = H_orig.detach()
            else:
                H_orig = self.cached_eval_h

        if objective == 'encode_clean':
            clean_zn, _ = self.get_node_edge_embeddings(x_hyp, H_orig, project=False)
            return clean_zn

        if self.training:
            # 【修改】根据策略选择视图生成方式
            if self.drop_strategy == 'critical':
                # 你的高级做法：获取关键度分数，生成非对称视图
                node_scores, edge_scores = self.get_critical_scores(x_hyp, H_orig)
                H1 = self.critical_aware_drop_incidence(H_orig, node_scores, edge_scores)
                H2 = self.hard_drop_incidence(H_orig, node_scores, edge_scores)
            elif self.drop_strategy == 'random':
                # 传统做法：纯随机均匀丢弃
                H1 = self.random_drop_incidence(H_orig)
                H2 = self.random_drop_incidence(H_orig)
            elif self.drop_strategy == 'topk':
                affinity = self.get_embedding_affinity(x_hyp)
                H1 = self.topk_drop_incidence(H_orig, affinity)
                node_scores, edge_scores = self.get_critical_scores(x_hyp, H_orig)
                H2 = self.hard_drop_incidence(H_orig, node_scores, edge_scores)
            else:
                raise ValueError(f"不支持的扰动策略: {self.drop_strategy}")

            # 特征扰动：注入双曲空间噪声 (这部分对两个策略都保留，或者你也可以把特征噪声也受策略控制)
            x_hyp_noisy = self.hyperbolic_feature_noise(
                x_hyp, noise_scale=self.feature_noise_scale)

            # 提取特征
            zn1, ze1 = self.get_node_edge_embeddings(x_hyp, H1)
            zn2, ze2 = self.get_node_edge_embeddings(x_hyp_noisy, H2)

            # 计算对比损失
            if objective == 'supcon':
                loss = self.contrast_loss.supervised_node_loss(
                    zn1, zn2, labels, train_idx, group_ids, cross_group,
                    coarse_labels, mask_same_major, pair_policy, patient_balance)
            elif objective == 'legacy':
                loss = self.contrast_loss(zn1, zn2, ze1, ze2, H=H_orig)
            else:
                raise ValueError(f"Unsupported contrastive objective: {objective}")
            snapshot = None
            if self.capture_topology:
                snapshot = {
                    'H_initial': H_input.detach().clone(),
                    'H_evolved': H1.detach().clone(),
                    'H_hard': H2.detach().clone(),
                    'node_embedding': zn1.detach().clone(),
                }
                self.last_snapshot = snapshot
            if return_clean:
                clean_zn, _ = self.get_node_edge_embeddings(
                    x_hyp, H_orig, project=False)
                return zn1, zn2, loss, snapshot, clean_zn
            return zn1, zn2, loss, snapshot
        else:
            # 推理阶段使用原始结构
            zn, _ = self.get_node_edge_embeddings(
                x_hyp, H_orig, project=not return_clean)
            if self.capture_topology:
                self.last_snapshot = {
                    'H_initial': H_input.detach().clone(),
                    'H_evolved': H_orig.detach().clone(),
                    'node_embedding': zn.detach().clone(),
                }
            return zn


class TransConvLayer(nn.Module):
    # 此处保持你原有的 TransConvLayer 实现不变
    def __init__(self, manifold, in_channels, out_channels, num_heads, use_weight=True, args=None):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.attention_type = args.attention_type

        self.Wk = nn.ModuleList()
        self.Wq = nn.ModuleList()
        for i in range(self.num_heads):
            self.Wk.append(HypLinear(self.manifold, self.in_channels, self.out_channels))
            self.Wq.append(HypLinear(self.manifold, self.in_channels, self.out_channels))

        if use_weight:
            self.Wv = nn.ModuleList()
            for i in range(self.num_heads):
                self.Wv.append(HypLinear(self.manifold, in_channels, out_channels))

        self.scale = nn.Parameter(torch.tensor([math.sqrt(out_channels)], requires_grad=True))
        self.bias = nn.Parameter(torch.zeros(()))
        self.norm_scale = nn.Parameter(torch.ones(()))
        self.v_map_mlp = nn.Linear(in_channels, out_channels, bias=True)
        self.power_k = args.power_k
        self.trans_heads_concat = args.trans_heads_concat

        if self.trans_heads_concat:
            self.final_linear = nn.Linear(out_channels * self.num_heads, out_channels, bias=True)

    def full_attention(self, qs, ks, vs, output_attn=False):
        att_weight = 2 + 2 * self.manifold.cinner(qs.transpose(0, 1), ks.transpose(0, 1))
        att_weight = att_weight / self.scale + self.bias
        att_weight = nn.Softmax(dim=-1)(att_weight)
        att_output = self.manifold.mid_point(vs.transpose(0, 1), att_weight)
        att_output = att_output.transpose(0, 1)
        att_output = self.manifold.mid_point(att_output)
        if output_attn:
            return att_output, att_weight
        else:
            return att_output

    @staticmethod
    def fp(x, p=2):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / norm_x_p) * x ** p

    def linear_focus_attention(self, hyp_qs, hyp_ks, hyp_vs, output_attn=False):
        qs = hyp_qs[..., 1:]
        ks = hyp_ks[..., 1:]
        v = hyp_vs[..., 1:]

        phi_qs = (F.relu(qs) + 1e-6) / self.norm_scale.abs()
        phi_ks = (F.relu(ks) + 1e-6) / self.norm_scale.abs()

        phi_qs = self.fp(phi_qs, p=self.power_k)
        phi_ks = self.fp(phi_ks, p=self.power_k)

        k_transpose_v = torch.einsum('nhm,nhd->hmd', phi_ks, v)
        numerator = torch.einsum('nhm,hmd->nhd', phi_qs, k_transpose_v)
        denominator = torch.einsum('nhd,hd->nh', phi_qs, torch.einsum('nhd->hd', phi_ks))
        denominator = denominator.unsqueeze(-1)
        attn_output = numerator / (denominator + 1e-6)

        vss = self.v_map_mlp(v)
        attn_output = attn_output + vss

        if self.trans_heads_concat:
            attn_output = self.final_linear(attn_output.reshape(-1, self.num_heads * self.out_channels))
        else:
            attn_output = attn_output.mean(dim=1)

        attn_output_time = ((attn_output ** 2).sum(dim=-1, keepdims=True) + self.manifold.k) ** 0.5
        attn_output = torch.cat([attn_output_time, attn_output], dim=-1)

        if output_attn:
            attention = torch.einsum('nhd,mhd->nmh', phi_qs, phi_ks)
            attention = attention / (denominator + 1e-6)
            attention = attention.mean(dim=-1)
            return attn_output, attention
        else:
            return attn_output

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        q_list, k_list, v_list = [], [], []
        for i in range(self.num_heads):
            q_list.append(self.Wq[i](query_input))
            k_list.append(self.Wk[i](source_input))
            if self.use_weight:
                v_list.append(self.Wv[i](source_input))
            else:
                v_list.append(source_input)

        query = torch.stack(q_list, dim=1)
        key = torch.stack(k_list, dim=1)
        value = torch.stack(v_list, dim=1)

        if output_attn:
            if self.attention_type == 'linear_focused':
                attention_output, attn = self.linear_focus_attention(query, key, value, output_attn)
            elif self.attention_type == 'full':
                attention_output, attn = self.full_attention(query, key, value, output_attn)
            else:
                raise NotImplementedError
        else:
            if self.attention_type == 'linear_focused':
                attention_output = self.linear_focus_attention(query, key, value)
            elif self.attention_type == 'full':
                attention_output = self.full_attention(query, key, value)

        if output_attn:
            return attention_output, attn
        else:
            return attention_output


class TransConv(nn.Module):
    # 此处保持你原有的 TransConv 实现不变
    def __init__(self, manifold_in, manifold_hidden, manifold_out, in_channels, hidden_channels, args=None):
        super().__init__()
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = args.trans_num_layers
        self.num_heads = args.trans_num_heads
        self.dropout_rate = args.trans_dropout
        self.use_bn = args.trans_use_bn
        self.residual = args.trans_use_residual
        self.use_act = args.trans_use_act
        self.use_weight = args.trans_use_weight

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.fcs.append(HypLinear(self.manifold_in, self.in_channels, self.hidden_channels, self.manifold_hidden))
        self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.add_pos_enc = args.add_positional_encoding
        self.positional_encoding = HypLinear(self.manifold_in, self.in_channels, self.hidden_channels,
                                             self.manifold_hidden)
        self.epsilon = torch.tensor([1.0], device=args.device)

        for i in range(self.num_layers):
            self.convs.append(
                TransConvLayer(self.manifold_hidden, self.hidden_channels, self.hidden_channels,
                               num_heads=self.num_heads,
                               use_weight=self.use_weight, args=args))
            self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.dropout = HypDropout(self.manifold_hidden, self.dropout_rate)
        self.activation = HypActivation(self.manifold_hidden, activation=F.relu)

        self.fcs.append(HypLinear(self.manifold_hidden, self.hidden_channels, self.hidden_channels, self.manifold_out))

    def forward(self, x_input):
        layer_ = []
        x = self.fcs[0](x_input, x_manifold='euc')
        if self.add_pos_enc:
            x_pos = self.positional_encoding(x_input, x_manifold='euc')
            x = self.manifold_in.mid_point(torch.stack((x, self.epsilon * x_pos), dim=1))

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        if self.dropout_rate > 0:
            x = self.dropout(x, training=self.training)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, x)
            if self.residual:
                x = self.manifold_in.mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)

        x = self.fcs[-1](x)
        return x

    def get_attentions(self, x_input):
        layer_, attentions = [], []
        x = self.fcs[0](x_input, x_manifold='euc')
        if self.add_pos_enc:
            x_pos = self.positional_encoding(x_input, x_manifold='euc')
            x = self.manifold_in.mid_point(torch.stack((x, self.epsilon * x_pos), dim=1))

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        if self.dropout_rate > 0:
            x = self.dropout(x, training=self.training)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.manifold_in.mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)


class HypFormer(nn.Module):
    def __init__(self, args):
        super().__init__()

        # 【新增】：讀取自適應曲率的開關狀態
        self.use_adaptive_curvature = getattr(args, 'use_adaptive_curvature', False)

        # 1. 根據開關決定是否初始化自适应曲率学习器
        if self.use_adaptive_curvature:
            self.k_in_learner = AdaptiveCurvatureLayer(args.k_in)
            self.k_out_learner = AdaptiveCurvatureLayer(args.k_out)
        else:
            self.k_in_learner = None
            self.k_out_learner = None

        # 2. 初始化流形 (这里先用 args 中的初始值，如果不開啟自適應，這就是最終固定的曲率)
        self.manifold_in = Lorentz(k=args.k_in)
        self.manifold_hidden = Lorentz(k=args.k_out)
        self.manifold_out = Lorentz(k=args.k_out)

        self.in_channels = args.in_channels
        self.hidden_channels = args.hidden_channels
        self.out_channels = args.out_channels
        self.use_graph = args.use_graph
        self.graph_weight = args.graph_weight
        self.decoder_type = args.decoder_type

        self.use_adv = getattr(args, 'use_cl', True)
        self.cl_mode = getattr(args, 'cl_mode', 'legacy')
        self.cl_weight = getattr(args, 'cl_weight', 0.1)
        self.cl_fusion_weight = getattr(args, 'cl_fusion_weight', 0.0)
        self.cl_aux_head = bool(getattr(args, 'cl_aux_head', 1))
        self.cl_consistency = bool(getattr(args, 'cl_consistency', 1))
        self.cl_consistency_weight = getattr(args, 'cl_consistency_weight', 0.2)
        self.cl_fine_labels = bool(getattr(args, 'cl_fine_labels', 0))
        self.cl_cross_patient = bool(getattr(args, 'cl_cross_patient', 0))
        self.cl_mask_same_major = bool(getattr(args, 'cl_mask_same_major', 0))
        self.cl_pair_policy = getattr(args, 'cl_pair_policy', 'diag_fixed')
        self.cl_patient_balance = bool(getattr(args, 'cl_patient_balance', 0))
        self.prototype_temperature = float(getattr(args, 'prototype_temperature', 0.2))
        self.hyperedge_policy = getattr(args, 'hyperedge_policy', 'global_learned')
        if self.hyperedge_policy not in ('global_learned', 'patient_state'):
            raise ValueError('Unknown hyperedge policy')
        self.last_supcon_loss = None
        self.last_prototype_stats = None
        self.last_consistency_loss = None

        # Construct the task backbone first so paired baseline/CL runs start
        # with identical task weights under the same seed.
        self.trans_conv = TransConv(self.manifold_in, self.manifold_hidden, self.manifold_out,
                                    self.in_channels, self.hidden_channels, args=args)
        self.graph_conv = GraphConv(self.in_channels, self.hidden_channels, args=args) if self.use_graph else None
        if self.decoder_type == 'euc':
            self.euc_ln = nn.LayerNorm(self.hidden_channels)
            self.decode_trans = nn.Linear(self.hidden_channels, self.out_channels)
            self.decode_graph = nn.Linear(self.hidden_channels, self.out_channels)
        elif self.decoder_type == 'hyp':
            self.decode_graph = HypLinear(self.manifold_out, self.hidden_channels, self.hidden_channels)
            self.decode_trans = HypCLS(self.manifold_out, self.hidden_channels, self.out_channels)
        else:
            raise NotImplementedError

        self.adv_module = None
        if self.use_adv or getattr(args, 'save_topology', 0):
            self.adv_module = HypHypergraphAdversarialModule(
                manifold=self.manifold_hidden,
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                k=getattr(args, 'cl_knn', 5),
                drop_rate=getattr(args, 'cl_drop_rate', 0.5),
                tau=getattr(args, 'cl_tau', 0.5),
                knn_metric=getattr(args, 'knn_metric', 'hyp'),
                drop_strategy=getattr(args, 'drop_strategy', 'critical'),
                cl_space=getattr(args, 'cl_space', 'hyp'),
                graph_build_method=getattr(args, 'graph_build_method', 'knn'),
                eps_threshold=getattr(args, 'eps_threshold', 1.5),
                walk_length=getattr(args, 'walk_length', 10),
                num_walks=5,
                capture_topology=getattr(args, 'save_topology', 0),
                graph_source=getattr(args, 'graph_source', 'input'),
                feature_noise_scale=getattr(args, 'cl_feature_noise', 0.1)
            )
        if self.use_adv:
            self.adapt_layer = HypLinear(self.manifold_hidden, self.hidden_channels, self.in_channels)
            if self.cl_mode in ('shared_supcon', 'patient_prototype'):
                for discriminator in (self.adv_module.node_discriminator,
                                      self.adv_module.edge_discriminator):
                    discriminator.requires_grad_(False)
                self.cl_input_norm = nn.LayerNorm(self.in_channels)
                self.cl_gate_logit = nn.Parameter(torch.tensor(-2.0))
                if self.cl_aux_head:
                    self.cl_head = nn.Sequential(
                        nn.LayerNorm(2 * self.in_channels),
                        nn.Linear(2 * self.in_channels, self.hidden_channels),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(self.hidden_channels, self.out_channels),
                    )
                    self.cl_head_gate = nn.Parameter(torch.tensor(-1.0))

    def _update_manifolds(self):
        """
        核心修复：使用 __dict__ 绕过 nn.Module 的检查直接赋值 Tensor
        如果未開啟自適應曲率，則直接 return，保持原有固定曲率。
        """
        # 【新增】：如果未使用自適應曲率，直接略過更新
        if not self.use_adaptive_curvature:
            return

        k_in = self.k_in_learner()
        k_out = self.k_out_learner()

        self.manifold_in.__dict__['k'] = k_in
        self.manifold_hidden.__dict__['k'] = k_out
        self.manifold_out.__dict__['k'] = k_out

    def _forward_task(self, x, edge_index, return_embedding=False):
        x1 = self.trans_conv(x)
        if self.use_graph:
            x2 = self.graph_conv(x, edge_index)
            if self.decoder_type == 'euc':
                euc_x = self.manifold_out.logmap0(x1)[..., 1:]
                euc_x = self.euc_ln(euc_x)
                out = (1 - self.graph_weight) * self.decode_trans(euc_x) + self.graph_weight * self.decode_graph(x2)
                if return_embedding:
                    predictive_embedding = torch.cat(
                        ((1 - self.graph_weight) * euc_x,
                         self.graph_weight * x2), dim=-1)
            elif self.decoder_type == 'hyp':
                z_graph_hyp = self.decode_graph(x2, x_manifold='euc')
                z = self.manifold_out.mid_point(
                    torch.stack([(1 - self.graph_weight) * x1, self.graph_weight * z_graph_hyp], dim=1))
                out = self.decode_trans(z)
            else:
                raise NotImplementedError
        else:
            if self.decoder_type == 'euc':
                euc_x = self.manifold_out.logmap0(x1)[..., 1:]
                euc_x = self.euc_ln(euc_x)
                out = self.decode_trans(euc_x)
            elif self.decoder_type == 'hyp':
                out = self.decode_trans(x1)
            else:
                raise NotImplementedError
        if return_embedding:
            if not self.use_graph or self.decoder_type != 'euc':
                raise ValueError('Patient prototype requires Euclidean graph decoder')
            return out, predictive_embedding
        return out

    def _shared_input(self, x, clean_embedding):
        adapted = self.adapt_layer(clean_embedding.squeeze(0), x_manifold='hyp')
        tangent = self.manifold_hidden.logmap0(adapted)[..., 1:]
        return x + torch.sigmoid(self.cl_gate_logit) * self.cl_input_norm(tangent), tangent

    def _shared_logits(self, task_input, tangent, raw_x, edge_index):
        logits = self._forward_task(task_input, edge_index)
        if self.cl_aux_head:
            auxiliary_input = torch.cat([raw_x, tangent], dim=-1)
            logits = logits + torch.sigmoid(self.cl_head_gate) * self.cl_head(auxiliary_input)
        return logits

    def _clean_hyper_embedding(self, dataset):
        x = dataset.graph['node_feat']
        incidence = None
        if self.hyperedge_policy == 'patient_state':
            if getattr(dataset, 'patient_codes', None) is None:
                raise ValueError('Patient-state hyperedges require patient IDs')
            incidence = getattr(dataset, '_patient_state_incidence', None)
            if incidence is None or incidence.device != x.device:
                from gse_patient_state_graph import patient_state_incidence
                incidence = patient_state_incidence(x, dataset.patient_codes)
                dataset._patient_state_incidence = incidence
        return self.adv_module(x, objective='encode_clean',
                               incidence_override=incidence)

    def forward(self, dataset, train_idx=None):
        self._update_manifolds()

        x = dataset.graph['node_feat']
        edge_index = dataset.graph['edge_index']

        if self.use_adv and self.cl_mode == 'patient_prototype':
            clean_embedding = self._clean_hyper_embedding(dataset)
            task_input, tangent = self._shared_input(x, clean_embedding)
            if self.training:
                if train_idx is None or getattr(dataset, 'patient_codes', None) is None:
                    raise ValueError('Patient prototype requires training indices and patient IDs')
                from gse_patient_modules import cross_patient_prototype_loss
                out, predictive_embedding = self._forward_task(
                    task_input, edge_index, return_embedding=True)
                if self.cl_aux_head:
                    auxiliary_input = torch.cat([x, tangent], dim=-1)
                    out = out + torch.sigmoid(self.cl_head_gate) * self.cl_head(auxiliary_input)
                prototype_loss, diagnostics = cross_patient_prototype_loss(
                    predictive_embedding, dataset.label.squeeze(1),
                    dataset.patient_codes, train_idx,
                    temperature=self.prototype_temperature,
                    state_labels=(dataset.cl_fine_labels if self.cl_fine_labels
                                  else None), expression_view=x,
                    hyper_view=tangent)
                self.last_prototype_stats = diagnostics
                self.last_supcon_loss = float(prototype_loss.detach())
                self.last_consistency_loss = 0.0
                return out, None, None, prototype_loss * self.cl_weight
            return self._shared_logits(task_input, tangent, x, edge_index)

        if self.use_adv and self.cl_mode == 'shared_supcon':
            if self.training:
                _, _, contrast_loss, _, clean_embedding = self.adv_module(
                    x, labels=(dataset.cl_fine_labels if self.cl_fine_labels
                               else dataset.label.squeeze(1)), train_idx=train_idx,
                    objective='supcon', return_clean=True,
                    group_ids=dataset.patient_codes if self.cl_cross_patient else None,
                    cross_group=self.cl_cross_patient,
                    coarse_labels=dataset.label.squeeze(1),
                    mask_same_major=self.cl_mask_same_major,
                    pair_policy=self.cl_pair_policy,
                    patient_balance=self.cl_patient_balance)
                task_input, tangent = self._shared_input(x, clean_embedding)
                out = self._shared_logits(task_input, tangent, x, edge_index)
                auxiliary_loss = out.new_zeros(())
                if self.cl_consistency and train_idx is not None:
                    noisy_x = x * (torch.rand_like(x) >= 0.15).to(x.dtype)
                    noisy_embedding = self.adv_module(noisy_x, objective='encode_clean')
                    noisy_input, noisy_tangent = self._shared_input(noisy_x, noisy_embedding)
                    keep = torch.rand(edge_index.size(1), device=edge_index.device) > 0.1
                    noisy_edge = edge_index[:, keep]
                    noisy_logits = self._shared_logits(noisy_input, noisy_tangent, noisy_x, noisy_edge)
                    target = F.softmax(out[train_idx].detach(), dim=-1)
                    consistency_loss = F.kl_div(
                        F.log_softmax(noisy_logits[train_idx], dim=-1), target,
                        reduction='batchmean')
                    noisy_supervision = F.cross_entropy(
                        noisy_logits[train_idx], dataset.label.squeeze(1)[train_idx])
                    auxiliary_loss = self.cl_consistency_weight * (consistency_loss + noisy_supervision)
                self.last_supcon_loss = contrast_loss.detach().item()
                self.last_consistency_loss = auxiliary_loss.detach().item()
                return out, None, None, contrast_loss * self.cl_weight + auxiliary_loss
            clean_embedding = self.adv_module(x, return_clean=True)
            task_input, tangent = self._shared_input(x, clean_embedding)
            return self._shared_logits(task_input, tangent, x, edge_index)

        if self.use_adv and self.training:
            if not self.cl_fusion_weight:
                out_orig = self._forward_task(x, edge_index)
            z1_hyp, z2_hyp, contrast_loss, _ = self.adv_module(x)
            contrast_loss = contrast_loss * self.cl_weight

            z1_hyp_adapted = self.adapt_layer(z1_hyp.squeeze(0), x_manifold='hyp')
            z1_euc = self.manifold_hidden.logmap0(z1_hyp_adapted)[..., 1:]

            N = x.size(0)
            if N > 5000:
                fused_view = z1_euc
                out1 = self._forward_task(z1_euc, edge_index)
                if self.cl_fusion_weight:
                    out_orig = self._forward_task(
                        x + self.cl_fusion_weight * torch.tanh(fused_view), edge_index)
                return out_orig, out1, None, contrast_loss
            else:
                z2_hyp_adapted = self.adapt_layer(z2_hyp.squeeze(0), x_manifold='hyp')
                z2_euc = self.manifold_hidden.logmap0(z2_hyp_adapted)[..., 1:]
                fused_view = (z1_euc + z2_euc) / 2
                out1 = self._forward_task(z1_euc, edge_index)
                out2 = self._forward_task(z2_euc, edge_index)
                if self.cl_fusion_weight:
                    out_orig = self._forward_task(
                        x + self.cl_fusion_weight * torch.tanh(fused_view), edge_index)
                return out_orig, out1, out2, contrast_loss
        else:
            if self.use_adv and self.cl_fusion_weight:
                z_hyp = self.adv_module(x)
                z_adapted = self.adapt_layer(z_hyp.squeeze(0), x_manifold='hyp')
                z_euc = self.manifold_hidden.logmap0(z_adapted)[..., 1:]
                x = x + self.cl_fusion_weight * torch.tanh(z_euc)
            return self._forward_task(x, edge_index)

    def get_attentions(self, x):
        return self.trans_conv.get_attentions(x)
