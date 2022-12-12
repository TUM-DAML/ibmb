from math import ceil
from typing import List

import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor

from data.data_utils import MyGraph
from dataloaders.BaseLoader import BaseLoader


# https://pytorch-geometric.readthedocs.io/en/2.0.3/_modules/torch_geometric/loader/neighbor_sampler.html#NeighborSampler
class NeighborSamplingLoader(torch.utils.data.DataLoader):

    def __init__(self,
                 graph: Data,
                 sizes: List[int],
                 node_idx,
                 **kwargs):

        adj = SparseTensor.from_edge_index(graph.edge_index, sparse_sizes=(graph.num_nodes, graph.num_nodes))
        adj = BaseLoader.normalize_adjmat(adj, 'sym')

        self.node_idx = node_idx
        self.batch_size = kwargs['batch_size']
        self.sizes = sizes
        self.adj_t = adj
        self.original_graph = graph

        super().__init__(node_idx, collate_fn=self.sample, **kwargs)

    def __getitem__(self, idx):
        return self.node_idx[idx]

    def __len__(self):
        return len(self.node_idx)

    @property
    def loader_len(self):
        return ceil(len(self.node_idx) / self.batch_size)

    def sample(self, batch):
        batch = torch.tensor(batch)
        adjs = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            adjs.append(adj_t)

        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]

        subg = MyGraph(x=self.original_graph.x[n_id],
                       y=self.original_graph.y[batch],
                       edge_index=adjs,
                       output_node_mask=torch.ones(len(batch), dtype=torch.bool))
        return subg
