import logging
from math import ceil

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected
from torch_sparse import SparseTensor

from dataloaders.utils import topk_ppr_matrix
from .BaseLoader import BaseLoader


class IBMBRandLoader(BaseLoader):

    def __init__(self, graph: Data,
                 output_indices: torch.LongTensor,
                 return_edge_index_type: str,
                 num_auxiliary_node_per_output: int,
                 alpha: float = 0.2,
                 eps: float = 1.e-4,
                 batch_size: int = 1,
                 **kwargs):

        self.out_aux_pairs = []

        assert is_undirected(graph.edge_index, num_nodes=graph.num_nodes), "Assume the graph to be undirected"
        self.output_indices = output_indices
        assert return_edge_index_type in ['adj', 'edge_index']
        self.return_edge_index_type = return_edge_index_type
        self.num_auxiliary_node_per_output = num_auxiliary_node_per_output
        self.batch_size = batch_size
        self.alpha = alpha
        self.eps = eps

        self.original_graph = graph  # need to cache the original graph
        self.adj = None
        self.create_node_wise_loader(graph)

        super().__init__(self.out_aux_pairs,
                         batch_size=batch_size, **kwargs)

    def create_node_wise_loader(self, graph: Data):
        logging.info("Start PPR calculation")
        _, neighbors = topk_ppr_matrix(graph.edge_index,
                                       graph.num_nodes,
                                       self.alpha,
                                       self.eps,
                                       self.output_indices,
                                       self.num_auxiliary_node_per_output)

        for p, n in zip(self.output_indices.numpy(), neighbors):
            self.out_aux_pairs.append((np.array([p]), n))

        self.indices_complete_check(self.out_aux_pairs, self.output_indices)

        if self.return_edge_index_type == 'adj':
            adj = SparseTensor.from_edge_index(graph.edge_index, sparse_sizes=(graph.num_nodes, graph.num_nodes))
            adj = self.normalize_adjmat(adj, normalization='rw')
            self.adj = adj

    def __getitem__(self, idx):
        return self.out_aux_pairs[idx]

    def __len__(self):
        return len(self.out_aux_pairs)

    @property
    def loader_len(self):
        return ceil(len(self.out_aux_pairs) / self.batch_size)

    def __collate__(self, data_list):
        out, aux = zip(*data_list)
        out = np.concatenate(out)
        aux = np.unique(np.concatenate(aux))  # DO UNION!
        mask = torch.from_numpy(np.in1d(aux, out))
        aux = torch.from_numpy(aux)

        subg = self.get_subgraph(aux,
                                 self.original_graph,
                                 self.return_edge_index_type,
                                 self.adj,
                                 output_node_mask=mask)
        return subg
