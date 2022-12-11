from typing import List, Union, Tuple, Optional

import numba
import numpy as np
import torch
from scipy.sparse import csr_matrix, coo_matrix
from torch_geometric.utils import coalesce
from torch_sparse import SparseTensor


def topk_ppr_matrix(edge_index: torch.Tensor,
                    num_nodes: int,
                    alpha: float,
                    eps: float,
                    output_node_indices: Union[np.ndarray, torch.LongTensor],
                    topk: int,
                    normalization='row') -> Tuple[csr_matrix, List[np.ndarray]]:
    """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""
    if isinstance(output_node_indices, torch.Tensor):
        output_node_indices = output_node_indices.numpy()

    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    edge_index_np = edge_index.cpu().numpy()

    _, indptr, out_degree = np.unique(edge_index_np[0],
                                      return_index=True,
                                      return_counts=True)
    indptr = np.append(indptr, len(edge_index_np[0]))

    neighbors, weights = calc_ppr_topk_parallel(indptr, edge_index_np[1], out_degree,
                                                alpha, eps, output_node_indices)

    # neighbors, weights = get_calc_ppr()(indptr, edge_index_np[1], out_degree, alpha, eps)

    ppr_matrix = construct_sparse(neighbors, weights, (len(output_node_indices), num_nodes))
    ppr_matrix = ppr_matrix.tocsr()

    neighbors = sparsify(neighbors, weights, topk)
    neighbors = [np.union1d(nei, pr) for nei, pr in zip(neighbors, output_node_indices)]

    if normalization == 'sym':
        # Assume undirected (symmetric) adjacency matrix
        deg_sqrt = np.sqrt(np.maximum(out_degree, 1e-12))
        deg_inv_sqrt = 1. / deg_sqrt

        row, col = ppr_matrix.nonzero()
        ppr_matrix.data = deg_sqrt[output_node_indices[row]] * ppr_matrix.data * deg_inv_sqrt[col]
    elif normalization == 'col':
        # Assume undirected (symmetric) adjacency matrix
        deg_inv = 1. / np.maximum(out_degree, 1e-12)

        row, col = ppr_matrix.nonzero()
        ppr_matrix.data = out_degree[output_node_indices[row]] * ppr_matrix.data * deg_inv[col]
    elif normalization == 'row':
        pass
    else:
        raise ValueError(f"Unknown PPR normalization: {normalization}")

    return ppr_matrix, neighbors


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return coo_matrix((np.concatenate(weights), (i, j)), shape)


# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/gdc.html#GDC.diffusion_matrix_approx
@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes) \
        -> Tuple[List[np.ndarray], List[np.ndarray]]:
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    for i in numba.prange(len(nodes)):
        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
        js[i] = np.array(j)
        vals[i] = np.array(val)
    return js, vals


@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]

    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)

    return list(p.keys()), list(p.values())


def sparsify(neighbors: List[np.ndarray], weights: List[np.ndarray], topk: int):
    new_neighbors = []
    for n, w in zip(neighbors, weights):
        idx_topk = np.argsort(w)[-topk:]
        new_neighbor = n[idx_topk]
        new_neighbors.append(new_neighbor)

    return new_neighbors


def get_partitions(edge_index: Union[torch.LongTensor, SparseTensor],
                   num_partitions: int,
                   indices: torch.LongTensor,
                   num_nodes: int,
                   output_weight: Optional[float] = None) -> List[torch.LongTensor]:
    """
    Graph partitioning using METIS.
    If output_weight is given, assign weights on output nodes.

    :param edge_index:
    :param num_partitions:
    :param indices:
    :param num_nodes:
    :param output_weight:
    :return:
    """

    assert isinstance(edge_index, (torch.LongTensor, SparseTensor)), f'Unsupported edge_index type {type(edge_index)}'
    if isinstance(edge_index, torch.LongTensor):
        edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes))

    if output_weight is not None and output_weight != 1:
        node_weight = torch.ones(num_nodes)
        node_weight[indices] = output_weight
    else:
        node_weight = None

    _, partptr, perm = edge_index.partition(num_parts=num_partitions,
                                            recursive=False,
                                            weighted=False,
                                            node_weight=node_weight)

    partitions = []
    for i in range(len(partptr) - 1):
        partitions.append(perm[partptr[i]: partptr[i + 1]])

    return partitions
