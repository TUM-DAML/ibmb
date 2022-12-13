from math import ceil
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from tqdm import tqdm

from .BaseLoader import BaseLoader


class IBMBReadyLoader(BaseLoader):
    """
    Batch-wise IBMB dataloader from paper Influence-Based Mini-Batching for Graph Neural Networks
    """

    def __init__(self, graph: Data,
                 batch_order: str,
                 return_edge_index_type: str,
                 batches: List[Tuple[np.ndarray, np.ndarray]],
                 adj: Optional[SparseTensor] = None,
                 sampler: Sampler = None,
                 **kwargs):

        self.subgraphs = []
        self.batch_wise_out_aux_pairs = batches

        self.original_graph = None
        if adj is None:
            adj = SparseTensor.from_edge_index(graph.edge_index, sparse_sizes=(graph.num_nodes, graph.num_nodes))
            adj = self.normalize_adjmat(adj, normalization='rw')
        adj = adj

        self.cache_data = kwargs['batch_size'] == 1
        self.batch_size = kwargs['batch_size']
        assert return_edge_index_type in ['adj', 'edge_index']
        self.return_edge_index_type = return_edge_index_type

        if len(self.batch_wise_out_aux_pairs) > 2:   # <= 2 order makes no sense
            ys = [graph.y[out].numpy() for out, _ in self.out_aux_pairs]
            sampler = self.define_sampler(batch_order,
                                          ys,
                                          graph.y.max().item() + 1)

        if self.cache_data:
            self.prepare_cache(graph, batches, adj)
        else:
            self.original_graph = graph  # need to cache the original graph
            self.adj = adj

        super().__init__(self.subgraphs if self.cache_data else self.batch_wise_out_aux_pairs, sampler=sampler, **kwargs)

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
        return len(self.batch_wise_out_aux_pairs)

    @property
    def loader_len(self):
        return ceil(len(self.batch_wise_out_aux_pairs) / self.batch_size)

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
