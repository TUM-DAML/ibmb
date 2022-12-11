import time

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_sparse import SparseTensor


def get_time():
    torch.cuda.synchronize()
    return time.time()


def normalize_adjmat(adj: [SparseTensor, csr_matrix],
                     normalization: str,
                     inplace: bool = False):

    assert normalization in ['sym', 'rw']

    if isinstance(adj, SparseTensor):
        if not inplace:
            adj = adj.clone()
        adj = adj.fill_value(1, dtype=torch.float32)
        degree = adj.sum(0)
    elif isinstance(adj, csr_matrix):
        if not inplace:
            adj = adj.copy()
        adj.data = np.ones_like(adj.data, dtype=np.float32)
        degree = adj.sum(0).A1

    degree[degree == 0.] = 1e-12
    deg_inv = 1 / degree

    if normalization == 'sym':
        deg_inv_sqrt = deg_inv ** 0.5
        if isinstance(adj, csr_matrix):
            adj = adj.multiply(deg_inv_sqrt.reshape(1, -1))
            adj = adj.multiply(deg_inv_sqrt.reshape(-1, 1))
        elif isinstance(adj, SparseTensor):
            adj = adj * deg_inv_sqrt.reshape(1, -1)
            adj = adj * deg_inv_sqrt.reshape(-1, 1)

    elif normalization == 'rw':
        if isinstance(adj, csr_matrix):
            adj = adj.multiply(deg_inv.reshape(-1, 1))
        elif isinstance(adj, SparseTensor):
            adj = adj * deg_inv.reshape(-1, 1)

    return adj


def kl_divergence(p: np.ndarray, q: np.ndarray):
    return (p * np.log(p / q)).sum()


def get_pair_wise_distance(ys: list, num_classes: int, dist_type: str = 'kl'):
    num_batches = len(ys)

    counts = np.zeros((num_batches, num_classes), dtype=np.int32)
    for i in range(num_batches):
        unique, count = np.unique(ys[i], return_counts=True)
        counts[i, unique] = count

    counts += 1
    counts = counts / counts.sum(1).reshape(-1, 1)
    pairwise_dist = np.zeros((num_batches, num_batches), dtype=np.float64)

    for i in range(0, num_batches - 1):
        for j in range(i + 1, num_batches):
            if dist_type == 'l1':
                pairwise_dist[i, j] = np.sum(np.abs(counts[i] - counts[j]))
            elif dist_type == 'kl':
                pairwise_dist[i, j] = kl_divergence(counts[i], counts[j]) + kl_divergence(counts[j], counts[i])
            else:
                raise ValueError

    pairwise_dist += pairwise_dist.T

#     # softmax
#     np.fill_diagonal(pairwise_dist, -1e5)
#     pairwise_dist = softmax(pairwise_dist, axis=1)
#     # ^ 2
#     pairwise_dist = pairwise_dist ** 2

    pairwise_dist += 1e-5   # for numerical stability
    np.fill_diagonal(pairwise_dist, 0.)

    return pairwise_dist


class MyGraph:
    def __init__(self, **kwargs):
        super().__init__()
        self.keys = kwargs.keys()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to(self, device, non_blocking=False):
        for k in self.keys:
            v = getattr(self, k)
            if isinstance(v, (torch.Tensor, SparseTensor)):
                setattr(self, k, v.to(device, non_blocking=non_blocking))
            if isinstance(v, list):
                if isinstance(v[0], (torch.Tensor, SparseTensor)):
                    setattr(self, k, [_v.to(device,non_blocking=non_blocking) for _v in v])
                else:
                    raise TypeError
        return self
