import logging
from math import ceil
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Sampler
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected
from torch_sparse import SparseTensor
from tqdm import tqdm

from dataloaders.utils import topk_ppr_matrix
from .BaseLoader import BaseLoader


class IBMBRandfixLoader(BaseLoader):

    def __init__(self, graph: Data,
                 batch_order: str,
                 output_indices: torch.LongTensor,
                 return_edge_index_type: str,
                 num_output_nodes_per_batch: int,
                 num_auxiliary_node_per_output: int,
                 alpha: float = 0.2,
                 eps: float = 1.e-4,
                 batch_size: int = 1,
                 sampler: Sampler = None,
                 **kwargs):

        self.out_aux_pairs = []
        self.subgraphs = []

        assert is_undirected(graph.edge_index, num_nodes=graph.num_nodes), "Assume the graph to be undirected"
        self.output_indices = output_indices.numpy()
        assert return_edge_index_type in ['adj', 'edge_index']
        self.return_edge_index_type = return_edge_index_type
        self.num_output_nodes_per_batch = num_output_nodes_per_batch
        self.num_auxiliary_node_per_output = num_auxiliary_node_per_output
        self.batch_size = batch_size
        self.alpha = alpha
        self.eps = eps

        self.cache_data = batch_size == 1
        if not self.cache_data:
            self.original_graph = graph
        else:
            self.original_graph = None
        self.adj = None
        self.create_node_wise_loader(graph)

        if len(self.out_aux_pairs) > 2:   # <= 2 order makes no sense
            ys = [graph.y[out].numpy() for out, _ in self.out_aux_pairs]
            sampler = self.define_sampler(batch_order,
                                          ys,
                                          graph.y.max().item() + 1)

        super().__init__(self.subgraphs if self.cache_data else self.out_aux_pairs, sampler=sampler, batch_size=batch_size, **kwargs)

    def create_node_wise_loader(self, graph: Data):
        logging.info("Start PPR calculation")
        _, neighbors = topk_ppr_matrix(graph.edge_index,
                                       graph.num_nodes,
                                       self.alpha,
                                       self.eps,
                                       self.output_indices,
                                       self.num_auxiliary_node_per_output)

        for p, n in zip(self.output_indices, neighbors):
            self.out_aux_pairs.append((np.array([p]), n))

        self.out_aux_pairs = self.merge_locality(self.out_aux_pairs)
        self.indices_complete_check(self.out_aux_pairs, self.output_indices)

        if self.return_edge_index_type == 'adj':
            adj = SparseTensor.from_edge_index(graph.edge_index, sparse_sizes=(graph.num_nodes, graph.num_nodes))
            adj = self.normalize_adjmat(adj, normalization='rw')
        else:
            adj = None

        if self.cache_data:
            self.prepare_cache(graph, self.out_aux_pairs, adj)
        else:
            if self.return_edge_index_type == 'adj':
                self.adj = adj

    def merge_locality(self, out_aux_pairs: List[Tuple[np.ndarray, np.ndarray]],):
        np.random.seed(2021)
        fetch_idx = np.random.permutation(len(out_aux_pairs))

        batches = []

        ps = []
        ns = []
        pbar = tqdm(fetch_idx)
        pbar.set_description("Creating batches for randomly fixed")
        for idx in pbar:
            p, n = out_aux_pairs[idx]
            ps.append(p)
            ns.append(n)

            if len(ps) == self.num_output_nodes_per_batch or idx == fetch_idx[-1]:
                out = np.concatenate(ps)
                aux = np.unique(np.concatenate(ns))
                batches.append((out, aux))

                ps = []
                ns = []

        return batches

    def prepare_cache(self, graph: Data,
                      batch_wise_out_aux_pairs: List[Tuple[np.ndarray, np.ndarray]],
                      adj: Optional[SparseTensor]):

        pbar = tqdm(batch_wise_out_aux_pairs)
        pbar.set_description("Caching data")
        for out, aux in pbar:
            mask = torch.from_numpy(np.in1d(aux, out))
            aux = torch.from_numpy(aux)
            subg = self.get_subgraph(aux,
                                     graph,
                                     self.return_edge_index_type,
                                     adj,
                                     output_node_mask=mask)
            self.subgraphs.append(subg)

    def __getitem__(self, idx):
        return self.subgraphs[idx] if self.cache_data else self.out_aux_pairs[idx]

    def __len__(self):
        return len(self.out_aux_pairs)

    @property
    def loader_len(self):
        return ceil(len(self.out_aux_pairs) / self.batch_size)

    def __collate__(self, data_list):
        if len(data_list) == 1 and isinstance(data_list[0], Data):
            return data_list[0]

        out, aux = zip(*data_list)
        out = np.concatenate(out)
        aux = np.unique(np.concatenate(aux))  # still need it to be overlapping
        mask = torch.from_numpy(np.in1d(aux, out))
        aux = torch.from_numpy(aux)

        subg = self.get_subgraph(aux,
                                 self.original_graph,
                                 self.return_edge_index_type,
                                 self.adj,
                                 output_node_mask=mask)
        return subg
