from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from scipy.sparse import csr_matrix, eye
from torch_geometric.data import Data
from torch_geometric.datasets import Reddit, Reddit2
from torch_sparse import SparseTensor

from neighboring.pernode_ppr_neighbor import topk_ppr_matrix
from . import normalize_adjmat
from .const import get_ppr_default


def check_consistence(mode: str,
                      neighbor_sampling: str, 
                      order: bool, 
                      sample: bool):
    assert mode in ['ppr', 'rand', 'randfix', 'part', 'clustergcn', 'n_sampling', 'rw_sampling', 'ladies', 'part+ppr', 'ppr_shadow']
    if mode in ['ppr', 'part', 'randfix', 'part+ppr']:
        assert (sample and order) == False
    else:
        assert (sample or order) == False
    
    if mode in ['ppr', 'rand', 'part+ppr']:
        assert neighbor_sampling in ['ppr', 'hk', 'pnorm']
    elif mode == 'part':
        assert neighbor_sampling in ['ladies', 'batch_hk', 'batch_ppr',]
    elif mode == 'clustergcn':
        assert (sample or order) == False
        
    if neighbor_sampling == 'ladies':
        assert mode == 'part'


def config_transform(dataset_name: str,
                     graphmodel: str,
                     len_split: Tuple[int, int, int],
                     mode: str,
                     neighbor_sampling: str,
                     num_nodes: int,
                     num_batches: List[int],
                     ppr_params: Optional[Dict],
                     ladies_params: Optional[Dict]):
    # need validation
    if ppr_params is None:
        ppr_params = get_ppr_default(dataset_name, graphmodel)

    merge_max_size = (num_nodes // num_batches[0] + 1,
                      num_nodes // num_batches[1] + 1,
                      num_nodes // num_batches[2] + 1)

    if 'ladies' in [mode, neighbor_sampling]:
        merge_max_size = ladies_params['sample_size']
    elif 'ppr' in [mode, neighbor_sampling]:
        if ppr_params['merge_max_size'] is not None:
            merge_max_size = ppr_params['merge_max_size']

    # make it evenly distributed for each split
    n1 = np.ceil(len_split[0] / ppr_params['primes_per_batch']).astype(int)
    n2 = np.ceil(len_split[1] / ppr_params['primes_per_batch'] / 2).astype(int)
    n3 = np.ceil(len_split[2] / ppr_params['primes_per_batch'] / 2).astype(int)
    primes_per_batch = (np.ceil(len_split[0] / n1).astype(int),
                        np.ceil(len_split[1] / n2).astype(int),
                        np.ceil(len_split[2] / n3).astype(int),)

    if isinstance(ppr_params['neighbor_topk'], int):
        neighbor_topk = [ppr_params['neighbor_topk']] * 3
    else:
        neighbor_topk = ppr_params['neighbor_topk']

    return merge_max_size, neighbor_topk, primes_per_batch, ppr_params


def load_data(dataset_name: str, 
              small_trainingset: float):
    """

    :param dataset_name:
    :param small_trainingset:
    :return:
    """
    if dataset_name.lower() in ['arxiv', 'products', 'papers100m']:
        dataset = PygNodePropPredDataset(name="ogbn-{:s}".format(dataset_name), root='./datasets')
        split_idx = dataset.get_idx_split()
        graph = dataset[0]
    elif dataset_name.lower().startswith('reddit'):
        if dataset_name == 'reddit2':
            dataset = Reddit2('./datasets/reddit2')
        elif dataset_name == 'reddit':
            dataset = Reddit('./datasets/reddit')
        else:
            raise ValueError
        graph = dataset[0]
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                     'valid': graph.val_mask.nonzero().reshape(-1),
                     'test': graph.test_mask.nonzero().reshape(-1)}
        graph.train_mask, graph.val_mask, graph.test_mask = None, None, None
    else:
        raise NotImplementedError
        
    train_indices = split_idx["train"].cpu().detach().numpy()
    
    if small_trainingset < 1:
        np.random.seed(2021)
        train_indices = np.sort(np.random.choice(train_indices, 
                                                 size=int(len(train_indices) * small_trainingset), 
                                                 replace=False, 
                                                 p=None))

    val_indices = split_idx["valid"].cpu().detach().numpy()
    test_indices = split_idx["test"].cpu().detach().numpy()
    return graph, (train_indices, val_indices, test_indices,)


# Todo: use pre-transform
class GraphPreprocess:
    def __init__(self,
                 self_loop: bool = True,
                 to_undirected: bool = True,
                 normalization: str = 'sym'):
        self.self_loop = self_loop
        self.to_undirected = to_undirected
        self.normalization = normalization

    def __call__(self, graph: Data):
        graph.y = graph.y.reshape(-1)
        graph.y = torch.nan_to_num(graph.y, nan=-1)
        graph.y = graph.y.to(torch.long)

        # add adj_t
        row, col = graph.edge_index.cpu().detach().numpy()
        graph.edge_index = None
        data = np.ones_like(row, dtype=np.bool_)
        adj = csr_matrix((data, (row, col)), shape=(graph.num_nodes, graph.num_nodes))

        if self.to_undirected:
            adj += adj.transpose()

        if self.self_loop:
            adj += eye(graph.num_nodes, dtype=np.bool_)

        adj = normalize_adjmat(adj, self.normalization, inplace=True)
        graph.adj_t = SparseTensor.from_scipy(adj)

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
            partitions.append(perm[partptr[i] : partptr[i + 1]].cpu().detach().numpy())

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
