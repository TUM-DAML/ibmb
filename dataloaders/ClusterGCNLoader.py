from math import ceil
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected
from torch_sparse import SparseTensor
from tqdm import tqdm

from dataloaders.utils import get_partitions
from .BaseLoader import BaseLoader


def cluster_loader(partitions: List[np.ndarray], prime_indices: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    loader = []
    for n, second_nodes in enumerate(tqdm(partitions)):
        primes_in_part, *_ = np.intersect1d(second_nodes, prime_indices, return_indices=True)
        if len(primes_in_part):
            loader.append((primes_in_part, second_nodes))

    return loader


class ClusterGCNLoader(BaseLoader):
    """
    Batch-wise IBMB dataloader from paper Influence-Based Mini-Batching for Graph Neural Networks
    """

    def __init__(self, graph: Data,
                 num_partitions: int,
                 output_indices: torch.LongTensor,
                 return_edge_index_type: str,
                 metis_output_weight: Optional[float] = None,
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
        self.metis_output_weight = metis_output_weight

        self.create_batch_wise_loader(graph)

        if not self.cache_data:
            self.original_graph = graph  # need to cache the original graph

        super().__init__(self.subgraphs if self.cache_data else self.batch_wise_out_aux_pairs, **kwargs)

    def create_batch_wise_loader(self, graph: Data):
        adj = SparseTensor.from_edge_index(graph.edge_index, sparse_sizes=(graph.num_nodes, graph.num_nodes))
        adj = self.normalize_adjmat(adj, normalization='rw')

        # graph partitioning
        partitions = get_partitions(adj,
                                    self.num_partitions,
                                    self.output_indices,
                                    graph.num_nodes,
                                    self.metis_output_weight)

        batch_wise_out_aux_pairs = cluster_loader([p.numpy() for p in partitions], self.output_indices.numpy())

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
