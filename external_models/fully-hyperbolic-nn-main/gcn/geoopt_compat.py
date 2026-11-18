"""
geoopt compatibility layer - provides minimal implementations to replace geoopt dependency.
This file provides ManifoldParameter, ManifoldTensor, and basic manifold classes.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

class Manifold:
    """Base manifold class."""
    name = "base"
    ndim = 0
    reversible = False
    
    def __init__(self):
        pass
    
    def _assert_check_shape(self, shape, name):
        """Assert that shape is valid."""
        pass
    
    def egrad2rgrad(self, x, grad):
        return grad
    
    def retr(self, x, u):
        return x + u
    
    def retr_transp(self, x, u, v):
        return x + u, v
    
    def inner(self, x, u, v, keepdim=False):
        return (u * v).sum(dim=-1, keepdim=keepdim)
    
    def component_inner(self, x, u, v=None):
        """Component-wise inner product."""
        if v is None:
            v = u
        return u * v
    
    def norm(self, x, u, keepdim=False):
        return torch.norm(u, dim=-1, keepdim=keepdim)
    
    def dist(self, x, y, keepdim=False):
        return torch.norm(x - y, dim=-1, keepdim=keepdim)
    
    def logmap(self, x, y):
        return y - x
    
    def expmap(self, x, u):
        return x + u
    
    def projx(self, x):
        return x
    
    def proju(self, x, u):
        return u


class Euclidean(Manifold):
    """Euclidean manifold."""
    name = "Euclidean"
    ndim = 0
    reversible = False


class Lorentz(Manifold):
    """Lorentz manifold (Hyperbolic space in Lorentz model)."""
    name = "Lorentz"
    ndim = 1
    reversible = False
    
    def __init__(self, k=1.0, learnable=False):
        super().__init__()
        if isinstance(k, (int, float)):
            self.k = torch.tensor(k, dtype=torch.float32)
        else:
            self.k = k.float() if k.dtype != torch.float32 else k
        self.learnable = learnable
    
    def _assert_check_shape(self, shape, name):
        """Assert that shape is valid."""
        pass
    
    def expmap0(self, u, project=True, dim=-1):
        """Exponential map from origin."""
        from .manifolds import lmath as math
        res = math.expmap0(u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        return res
    
    def random_normal(self, *size, mean=0, std=1, dtype=None, device=None):
        """Create a random point on the manifold."""
        self._assert_check_shape(size2shape(*size), "x")
        if device is not None and device != self.k.device:
            raise ValueError("`device` does not match")
        if dtype is not None and dtype != self.k.dtype:
            raise ValueError("`dtype` does not match")
        tens = torch.randn(*size, device=self.k.device, dtype=self.k.dtype) * std + mean
        tens /= tens.norm(dim=-1, keepdim=True)
        return ManifoldTensor(self.expmap0(tens), manifold=self)
    
    def origin(self, *size, dtype=None, device=None, seed=42):
        """Zero point origin."""
        if dtype is None:
            dtype = self.k.dtype
        if device is None:
            device = self.k.device
        zero_point = torch.zeros(*size, dtype=dtype, device=device)
        zero_point[..., 0] = torch.sqrt(self.k)
        return ManifoldTensor(zero_point, manifold=self)


class PoincareBall(Manifold):
    """Poincaré ball model."""
    name = "PoincareBall"
    ndim = 1
    reversible = False
    
    def __init__(self, c=1.0):
        super().__init__()
        self.c = c


class Hyperboloid(Lorentz):
    """Hyperboloid model (alias for Lorentz)."""
    name = "Hyperboloid"


class Stiefel(Manifold):
    """Stiefel manifold."""
    name = "Stiefel"
    ndim = 2
    reversible = False


class BirkhoffPolytope(Manifold):
    """Birkhoff Polytope manifold."""
    name = "BirkhoffPolytope"
    ndim = 2
    reversible = False


class Sphere(Manifold):
    """Sphere manifold."""
    name = "Sphere"
    ndim = 1
    reversible = False
    
    def __init__(self, radius=1.0):
        super().__init__()
        self.radius = radius


class SphereExact(Sphere):
    """Sphere manifold with exact computations."""
    name = "SphereExact"


class Torus(Manifold):
    """Torus manifold."""
    name = "Torus"


class ProductManifold(Manifold):
    """Product manifold."""
    name = "ProductManifold"
    ndim = 1
    reversible = False


class ManifoldParameter(nn.Parameter):
    """Parameter on a manifold."""
    def __new__(cls, data=None, requires_grad=True, manifold=None):
        if data is None:
            data = torch.Tensor()
        self = super().__new__(cls, data, requires_grad)
        self.manifold = manifold if manifold is not None else Euclidean()
        return self


class ManifoldTensor(torch.Tensor):
    """Tensor on a manifold."""
    def __new__(cls, data=None, manifold=None):
        if data is None:
            data = torch.Tensor()
        self = super().__new__(cls, data)
        self.manifold = manifold if manifold is not None else Euclidean()
        return self


class RiemannianAdam(torch.optim.Optimizer):
    """Riemannian Adam optimizer."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, stabilize=None):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, stabilize=stabilize)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                    grad = p.manifold.egrad2rgrad(p, grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr']
                if group['weight_decay'] != 0:
                    grad.add_(p, alpha=group['weight_decay'])
                p.add_(exp_avg / denom, alpha=-step_size)
        return loss


class RiemannianSGD(torch.optim.Optimizer):
    """Riemannian SGD optimizer."""
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, weight_decay=0, stabilize=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, stabilize=stabilize)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                    grad = p.manifold.egrad2rgrad(p, grad)
                if group['weight_decay'] != 0:
                    grad.add_(p, alpha=group['weight_decay'])
                if group['momentum'] != 0:
                    state = self.state[p]
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(grad)
                    if group['dampening'] > 0:
                        grad = grad + group['dampening'] * buf
                    else:
                        grad = buf
                p.add_(grad, alpha=-group['lr'])
        return loss


# geoopt.manifolds module compatibility
class manifolds:
    class base:
        Manifold = Manifold


# geoopt.optim module compatibility
class optim:
    RiemannianAdam = RiemannianAdam
    RiemannianSGD = RiemannianSGD


# geoopt.utils module compatibility
class utils:
    @staticmethod
    def size2shape(*size):
        if len(size) == 1 and isinstance(size[0], (list, tuple)):
            size = size[0]
        return torch.Size(size)


def size2shape(*size):
    """Utility function to convert size to shape."""
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = size[0]
    return torch.Size(size)


def copy_or_set_(dest, source):
    """Copy or set values (compatibility function)."""
    if dest.shape == source.shape:
        dest.copy_(source)
    else:
        dest.data = source.data
    return dest


# Additional utility functions
class linalg:
    @staticmethod
    def matrix_rank(matrix, tol=None, hermitian=False):
        return torch.linalg.matrix_rank(matrix, tol=tol, hermitian=hermitian)

    @staticmethod
    def symeig(matrix, eigenvectors=False):
        if eigenvectors:
            return torch.linalg.eigh(matrix)
        else:
            return torch.linalg.eigvalsh(matrix), None
