from math import ceil
from typing import Optional, Union, List, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected
from torch_sparse import SparseTensor
from tqdm import tqdm

from dataloaders.utils import get_partitions
from .BaseLoader import BaseLoader


def ppr_power_method(adj: SparseTensor,
                     batch: List[Union[np.ndarray, torch.LongTensor]],
                     topk: int,
                     num_iter: int,
                     alpha: float) -> List[np.ndarray]:
    """
    PPR power iteration.

    :param adj:
    :param batch:
    :param topk:
    :param num_iter:
    :param alpha:
    :return:
    """
    topk_neighbors = []
    logits = torch.zeros(adj.size(0), len(batch), device=adj.device())  # each column contains a set of output nodes
    for i, tele_set in enumerate(batch):
        logits[tele_set, i] = 1. / len(tele_set)

    new_logits = logits.clone()
    for i in range(num_iter):
        new_logits = adj @ new_logits * (1 - alpha) + alpha * logits

    inds = new_logits.argsort(0)
    nonzeros = (new_logits > 0).sum(0)
    nonzeros = torch.minimum(nonzeros, torch.tensor([topk], dtype=torch.int64, device=adj.device()))
    for i in range(new_logits.shape[1]):
        topk_neighbors.append(inds[-nonzeros[i]:, i].cpu().numpy())

    return topk_neighbors


def create_batchwise_out_aux_pairs(adj: SparseTensor,
                                   partitions: List[Union[torch.LongTensor, np.ndarray]],
                                   prime_indices: Union[torch.LongTensor, np.ndarray],
                                   topk: int,
                                   num_outnodeset_per_batch: int = 50,
                                   alpha: float = 0.2,
                                   ppr_iterations: int = 50) -> List[Tuple[np.ndarray, np.ndarray]]:
    """

    :param adj:
    :param partitions:
    :param prime_indices:
    :param topk:
    :param num_outnodeset_per_batch:
    :param alpha:
    :param ppr_iterations:
    :return:
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if isinstance(prime_indices, torch.Tensor):
        prime_indices = prime_indices.cpu().numpy()

    adj = adj.to(device)

    cur_output_nodes = []
    loader = []

    pbar = tqdm(range(len(partitions)))
    pbar.set_description("Processing topic-sensitive PPR batches")
    for n in pbar:
        part = partitions[n]
        if isinstance(part, torch.Tensor):
            part = part.cpu().numpy()

        primes_in_part, *_ = np.intersect1d(part, prime_indices, assume_unique=True, return_indices=True)
        if len(primes_in_part):  # There ARE output nodes in this partition
            cur_output_nodes.append(primes_in_part)

        # accumulate enough output nodes for a batch, to make good use of GPU memory
        if len(cur_output_nodes) >= num_outnodeset_per_batch or n == len(partitions) - 1:
            topk_neighbors = ppr_power_method(adj, cur_output_nodes, topk, ppr_iterations, alpha)
            for i in range(len(cur_output_nodes)):
                # force output nodes to be aux nodes
                auxiliary_nodes = np.union1d(cur_output_nodes[i], topk_neighbors[i])
                loader.append((cur_output_nodes[i], auxiliary_nodes))
            cur_output_nodes = []

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return loader


class IBMBBatchLoader(BaseLoader):
    """
    Batch-wise IBMB dataloader from paper Influence-Based Mini-Batching for Graph Neural Networks
    """

    def __init__(self, graph: Data,
                 batch_order: str,
                 num_partitions: int,
                 output_indices: torch.LongTensor,
                 return_edge_index_type: str,
                 batch_expand_ratio: Optional[float] = 1.,
                 metis_output_weight: Optional[float] = None,
                 num_outnodeset_per_batch: Optional[int] = 50,
                 alpha: float = 0.2,
                 approximate_ppr_iterations: int = 50,
                 sampler: Sampler = None,
                 **kwargs):

        self.subgraphs = []
        self.batch_wise_out_aux_pairs = []

        self.original_graph = None
        self.adj = None

        assert is_undirected(graph.edge_index, num_nodes=graph.num_nodes), "Assume the graph to be undirected"
        self.cache_data = kwargs['batch_size'] == 1
        self._batchsize = kwargs['batch_size']
        self.num_partitions = num_partitions
        self.output_indices = output_indices
        assert return_edge_index_type in ['adj', 'edge_index']
        self.return_edge_index_type = return_edge_index_type
        self.batch_expand_ratio = batch_expand_ratio
        self.metis_output_weight = metis_output_weight
        self.num_outnodeset_per_batch = num_outnodeset_per_batch
        self.alpha = alpha
        self.approximate_ppr_iterations = approximate_ppr_iterations

        self.create_batch_wise_loader(graph)

        if len(self.batch_wise_out_aux_pairs) > 2:   # <= 2 order makes no sense
            ys = [graph.y[out].numpy() for out, _ in self.batch_wise_out_aux_pairs]
            sampler = self.define_sampler(batch_order,
                                          ys,
                                          graph.y.max().item() + 1)

        if not self.cache_data:
            self.original_graph = graph  # need to cache the original graph

        super().__init__(self.subgraphs if self.cache_data else self.batch_wise_out_aux_pairs, sampler=sampler, **kwargs)

    def create_batch_wise_loader(self, graph: Data):
        adj = SparseTensor.from_edge_index(graph.edge_index, sparse_sizes=(graph.num_nodes, graph.num_nodes))
        adj = self.normalize_adjmat(adj, normalization='rw')

        # graph partitioning
        partitions = get_partitions(adj,
                                    self.num_partitions,
                                    self.output_indices,
                                    graph.num_nodes,
                                    self.metis_output_weight)

        # get output - auxiliary node pairs
        topk = ceil(self.batch_expand_ratio * graph.num_nodes / self.num_partitions)
        batch_wise_out_aux_pairs = create_batchwise_out_aux_pairs(adj,
                                                                  partitions,
                                                                  self.output_indices,
                                                                  topk,
                                                                  self.num_outnodeset_per_batch,
                                                                  self.alpha,
                                                                  self.approximate_ppr_iterations)

        self.indices_complete_check(batch_wise_out_aux_pairs, self.output_indices)
        self.batch_wise_out_aux_pairs = batch_wise_out_aux_pairs

        if self.cache_data:
            self.prepare_cache(graph, batch_wise_out_aux_pairs, adj)
        else:
            if self.return_edge_index_type == 'adj':
                self.adj = adj

    def prepare_cache(self, graph: Data,
                      batch_wise_out_aux_pairs: List[Tuple[np.ndarray, np.ndarray]],
                      adj: Optional[SparseTensor]):

        pbar = tqdm(batch_wise_out_aux_pairs)
        pbar.set_description(f"Caching data with type {self.return_edge_index_type}")

        if self.return_edge_index_type == 'adj':
            assert adj is not None, "Trying to cache adjacency matrix, got None type."

        for out, aux in pbar:
            mask = torch.from_numpy(np.in1d(aux, out))

            if isinstance(aux, np.ndarray):
                aux = torch.from_numpy(aux)

            subg = self.get_subgraph(aux, graph, self.return_edge_index_type, adj, output_node_mask=mask)
            self.subgraphs.append(subg)

    def __getitem__(self, idx):
        return self.subgraphs[idx] if self.cache_data else self.batch_wise_out_aux_pairs[idx]

    def __len__(self):
        assert self.num_partitions == len(self.batch_wise_out_aux_pairs)
        return self.num_partitions

    @property
    def loader_len(self):
        return ceil(self.num_partitions / self._batchsize)

    def __collate__(self, data_list):
        if len(data_list) == 1 and isinstance(data_list[0], Data):
            return data_list[0]

        out, aux = zip(*data_list)
        out = np.concatenate(out)
        aux = np.unique(np.concatenate(aux))
        mask = torch.from_numpy(np.in1d(aux, out))
        aux = torch.from_numpy(aux)

        subg = self.get_subgraph(aux,
                                 self.original_graph,
                                 self.return_edge_index_type,
                                 self.adj,
                                 output_node_mask=mask)
        return subg
