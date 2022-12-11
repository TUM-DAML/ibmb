from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from scipy.sparse import csr_matrix, eye
from torch_geometric.data import Data
from torch_geometric.datasets import Reddit, Reddit2
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch_sparse import SparseTensor

from neighboring.pernode_ppr_neighbor import topk_ppr_matrix
from . import normalize_adjmat
from .const import get_ppr_default


def check_consistence(mode: str, batch_order: str):
    assert mode in ['ppr', 'rand', 'randfix', 'part',
                    'clustergcn', 'n_sampling', 'rw_sampling', 'ladies', 'ppr_shadow']
    if mode in ['ppr', 'part', 'randfix',]:
        assert batch_order in ['rand', 'sample', 'order']
    else:
        assert batch_order == 'rand'


def load_data(dataset_name: str,
              small_trainingset: float,
              pretransform):
    """

    :param dataset_name:
    :param small_trainingset:
    :param pretransform:
    :return:
    """
    if dataset_name.lower() in ['arxiv', 'products', 'papers100m']:
        dataset = PygNodePropPredDataset(name="ogbn-{:s}".format(dataset_name),
                                         root='./datasets',
                                         pre_transform=pretransform)
        split_idx = dataset.get_idx_split()
        graph = dataset[0]
    elif dataset_name.lower().startswith('reddit'):
        if dataset_name == 'reddit2':
            dataset = Reddit2('./datasets/reddit2', pre_transform=pretransform)
        elif dataset_name == 'reddit':
            dataset = Reddit('./datasets/reddit', pre_transform=pretransform)
        else:
            raise ValueError
        graph = dataset[0]
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                     'valid': graph.val_mask.nonzero().reshape(-1),
                     'test': graph.test_mask.nonzero().reshape(-1)}
        graph.train_mask, graph.val_mask, graph.test_mask = None, None, None
    else:
        raise NotImplementedError

    train_indices = split_idx["train"].numpy()

    if small_trainingset < 1:
        np.random.seed(2021)
        train_indices = np.sort(np.random.choice(train_indices,
                                                 size=int(len(train_indices) * small_trainingset),
                                                 replace=False,
                                                 p=None))

    train_indices = torch.from_numpy(train_indices)

    val_indices = split_idx["valid"]
    test_indices = split_idx["test"]
    return graph, (train_indices, val_indices, test_indices,)


class GraphPreprocess:
    def __init__(self,
                 self_loop: bool = True,
                 transform_to_undirected: bool = True):
        self.self_loop = self_loop
        self.to_undirected = transform_to_undirected

    def __call__(self, graph: Data):
        graph.y = graph.y.reshape(-1)
        graph.y = torch.nan_to_num(graph.y, nan=-1)
        graph.y = graph.y.to(torch.long)

        if self.self_loop:
            edge_index, _ = add_remaining_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
        else:
            edge_index = graph.edge_index

        if self.to_undirected:
            edge_index = to_undirected(edge_index, num_nodes=graph.num_nodes)

        graph.edge_index = edge_index
        return graph


def graph_preprocess(graph: Data,
                     self_loop: bool = True,
                     to_undirected: bool = True,
                     normalization: str = 'sym'):
    """

    :param graph:
    :param self_loop:
    :param to_undirected:
    :param normalization:
    :return:
    """
    if graph.y.dim() > 1:
        graph.y = graph.y.reshape(-1)
    if torch.isnan(graph.y).any():
        graph.y = torch.nan_to_num(graph.y, nan=-1)
    if graph.y.dtype != torch.int64:
        graph.y = graph.y.to(torch.long)

    row, col = graph.edge_index.cpu().detach().numpy()
    graph.edge_index = None
    data = np.ones_like(row, dtype=np.bool_)
    adj = csr_matrix((data, (row, col)), shape=(graph.num_nodes, graph.num_nodes))

    if to_undirected:
        adj += adj.transpose()

    if self_loop:
        adj += eye(graph.num_nodes, dtype=np.bool_)

    adj = normalize_adjmat(adj, normalization, inplace=True)
    graph.adj_t = SparseTensor.from_scipy(adj)


def get_partitions(mode: str,
                   mat: SparseTensor,
                   indices: np.ndarray,
                   num_parts: int,
                   force: bool = False) -> list:
    partitions = None
    if mode in ['part', 'clustergcn', 'part+ppr'] or force:
        if mode == 'part+ppr':
            node_weight = torch.ones(mat.sizes()[0])
            node_weight[indices] = 100000
        else:
            node_weight = None
        _, partptr, perm = mat.partition(num_parts=num_parts, recursive=False, weighted=False, node_weight=node_weight)

        partitions = []
        for i in range(len(partptr) - 1):
            partitions.append(perm[partptr[i]: partptr[i + 1]].cpu().detach().numpy())

    return partitions


def get_ppr_mat(mode: str,
                neighbor_sampling: str,
                prime_indices: np.ndarray,
                scipy_adj: csr_matrix,
                topk=256,
                eps=None,
                alpha=0.05) -> csr_matrix:
    ppr_mat = None
    if 'ppr' in [mode, neighbor_sampling]:
        # if too many prime nodes, we don't need many pairs
        if eps is None:
            eps = 1e-4 if (scipy_adj.nnz / len(prime_indices) < 100) else 1e-5
        ppr_mat = topk_ppr_matrix(scipy_adj, alpha, eps, prime_indices, topk=topk, normalization='sym')

    return ppr_mat
