"""Base model class."""

import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1, MarginLoss


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:

            # === [核心修复] 适度压制初始特征，但不要过度 ===
            # Twitch等高维特征网络，在进入双曲空间前需要缩小尺度
            # 但0.1太小了，改为0.3保持更多信息
            x = x * 0.3
            # ==============================================================

            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
            if self.manifold.name == 'Lorentz':
                x = self.manifold.expmap0(x)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        self.margin = args.margin
        if args.n_classes > 2:
            # === [核心修复] 使用 weighted 而非 micro，避免与 accuracy 重复 ===
            # micro-average 在多分类下等同于 accuracy，无法提供额外信息
            # weighted 考虑类别不平衡，macro 平等对待所有类别
            self.f1_average = 'weighted'
        else:
            self.f1_average = 'binary'
        self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return output[idx]

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        labels = data['labels'][idx]

        if self.manifold_name == 'Lorentz':
            batch_size = output.size(0)
            n_classes = output.size(1)

            # ==================== [高级细化 Debug 1: 距离分布诊断] ====================
            if split == 'train' and torch.rand(1).item() < 0.05:
                print(f"\n[Debug 距离矩阵] Max: {output.max().item():.4f}, Min: {output.min().item():.4f}, Mean: {output.mean().item():.4f}")
                print(f"[Debug 距离矩阵 Std]: {output.std().item():.4f}")
                if torch.isnan(output).any():
                    print("[!] 严重警告: Output 已经出现了 NaN！")
            # =========================================================================

            # === [核心修复] 重新设计Margin Loss的计算逻辑 ===
            # output现在是负距离（越大越相似/距离越小）
            # 对于正确类别，我们希望output更大（距离更小）
            # 对于错误类别，我们希望output更小（距离更大）
            
            positive_scores = output.gather(1, labels.unsqueeze(-1)).squeeze(-1)

            mask = torch.ones_like(output, dtype=torch.bool)
            mask.scatter_(1, labels.unsqueeze(-1), 0)
            negative_scores = output.masked_select(mask).view(batch_size, n_classes - 1)

            # ==================== [高级细化 Debug 2: 评分差值诊断] ====================
            if split == 'train' and torch.rand(1).item() < 0.05:
                pos_mean = positive_scores.mean().item()
                neg_mean = negative_scores.mean().item()
                print(f"[Debug 正负类分数] 正确类平均: {pos_mean:.4f} (应该大), 错误类平均: {neg_mean:.4f} (应该小)")
                print(f"[Debug 差距]: {pos_mean - neg_mean:.4f} (应该 > margin={self.margin})")
            # =========================================================================

            positive_expanded = positive_scores.unsqueeze(1).expand_as(negative_scores)

            # === [核心修正] Margin Loss的正确形式 ===
            # 我们希望: positive_scores > negative_scores + margin
            # 即: positive_scores - negative_scores > margin
            # loss = ReLU(margin - (positive_scores - negative_scores))
            #      = ReLU(margin - positive_scores + negative_scores)
            loss_tensor = F.relu(self.margin - positive_expanded + negative_scores)
            loss = loss_tensor.mean()

            # ==================== [高级细化 Debug 3: 损失震荡诊断] ====================
            if split == 'train' and (torch.isnan(loss) or loss.item() > 10.0):
                print(f"[!] 警告: 发现异常高损: {loss.item():.4f}")
            # =========================================================================

            # === [核心修复] 预测时使用argmax（因为output是负距离，越大越相似）===
            preds = output.max(1)[1]
            acc, f1 = acc_f1(preds, labels, average=self.f1_average)
        else:
            # === [核心修复] 对于非Lorentz流形，使用带权重的交叉熵 ===
            loss = F.cross_entropy(output, labels, weight=self.weights if hasattr(self, 'weights') else None)
            acc, f1 = acc_f1(output, labels, average=self.f1_average)

        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
        self.loss = MarginLoss(args.margin)

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        return -sqdist

    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        preds = torch.stack([pos_scores, neg_scores], dim=-1)
        loss = self.loss(preds)

        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        
        grad_norms = [torch.norm(p.grad).detach().item() for p in self.parameters() if p.grad is not None]
        if grad_norms:
            metrics = {'loss': loss, 'roc': roc, 'ap': ap, 'grad_norm': sum(grad_norms)}
        else:
            metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

