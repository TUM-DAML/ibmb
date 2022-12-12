import os.path as osp
from math import ceil
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from tqdm import tqdm

from dataloaders.BaseLoader import BaseLoader


class SaintRWTrainSampler(BaseLoader):

    def __init__(self,
                 graph: Data,
                 output_indices: torch.LongTensor,
                 return_edge_index_type: str,
                 num_nodes: int,
                 batch_size: int,
                 walk_length: int,
                 num_steps: int = 1,
                 sample_coverage: int = 0,
                 save_dir: Optional[str] = None,
                 log: bool = True,
                 **kwargs):

        self.walk_length = walk_length
        self.num_steps = num_steps
        self.__batch_size__ = batch_size
        self.sample_coverage = sample_coverage
        self.log = log

        self.N = num_nodes
        self.E = graph.num_edges

        self.adj = SparseTensor.from_edge_index(graph.edge_index, sparse_sizes=(graph.num_nodes, graph.num_nodes))
        self.adj = self.normalize_adjmat(self.adj, normalization='rw')
        self.edge_weight = self.adj.storage.value()
        self.adj.set_value_(torch.arange(self.E))

        self.original_graph = graph
        self.output_indices = output_indices

        assert return_edge_index_type in ['adj', 'edge_index']
        self.return_edge_index_type = return_edge_index_type

        super().__init__(self, batch_size=1, **kwargs)

        if self.sample_coverage > 0:
            path = osp.join(save_dir or '', self.__filename__)
            if save_dir is not None and osp.exists(path):  # pragma: no cover
                self.node_norm, self.edge_norm = torch.load(path)
            else:
                self.node_norm, self.edge_norm = self.__compute_norm__()
                if save_dir is not None:  # pragma: no cover
                    torch.save((self.node_norm, self.edge_norm), path)

    @property
    def __filename__(self):
        return (f'{self.N}_{self.walk_length}_'
                f'{self.sample_coverage}.pt')

    def __len__(self):
        return self.num_steps

    def loader_len(self):
        return self.num_steps

    def __sample_nodes__(self, batch_size):
        start = torch.randint(0, self.N, (batch_size,), dtype=torch.long)
        node_idx = self.adj.random_walk(start.flatten(), self.walk_length)
        return node_idx.view(-1)

    def __getitem__(self, idx):
        node_idx = self.__sample_nodes__(self.__batch_size__).unique()
        adj, edge_mask = self.adj.saint_subgraph(node_idx)
        return node_idx, adj

    def __collate__(self, data_list):
        assert len(data_list) == 1
        node_idx, adj = data_list[0]

        mask = torch.from_numpy(np.in1d(node_idx.cpu().numpy(), self.output_indices))

        if self.sample_coverage > 0:
            node_norm = self.node_norm[node_idx]
            node_norm = node_norm[mask]
            node_norm /= node_norm.sum()
        else:
            node_norm = None

        _, _, edge_idx = adj.coo()
        if self.return_edge_index_type == 'adj':
            if self.sample_coverage > 0:
                adj.set_value_(self.edge_norm[edge_idx] * self.edge_weight[edge_idx], layout='csr')
            else:
                adj.set_value_(self.edge_weight[edge_idx], layout='csr')
        else:
            adj = torch.vstack([adj.storage.row(), adj.storage.col()])

        graph = Data(x=self.original_graph.x[node_idx],
                     y=self.original_graph.y[node_idx],
                     edge_index=adj,
                     output_node_mask=mask,
                     node_norm=node_norm)

        return graph

    def __compute_norm__(self):
        node_count = torch.zeros(self.N, dtype=torch.float)
        edge_count = torch.zeros(self.E, dtype=torch.float)

        loader = torch.utils.data.DataLoader(self, batch_size=200,
                                             collate_fn=lambda x: x,
                                             num_workers=self.num_workers)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.N * self.sample_coverage)
            pbar.set_description('Compute GraphSAINT normalization')

        num_samples = 0
        total_sampled_nodes = 0
        while total_sampled_nodes < self.N * self.sample_coverage:
            for data in loader:
                for node_idx, adj in data:
                    edge_idx = adj.storage.value()
                    node_count[node_idx] += 1
                    edge_count[edge_idx] += 1
                    total_sampled_nodes += node_idx.size(0)

                    if self.log:  # pragma: no cover
                        pbar.update(node_idx.size(0))
            num_samples += self.num_steps

        if self.log:  # pragma: no cover
            pbar.close()

        row, _, edge_idx = self.adj.coo()
        t = torch.empty_like(edge_count).scatter_(0, edge_idx, node_count[row])
        edge_norm = (t / edge_count)
        edge_norm[edge_norm == float('inf')] = 0.1
        edge_norm[torch.isnan(edge_norm)] = 0.1

        node_count[node_count == 0] = 0.1
        node_norm = num_samples / node_count / self.N

        return node_norm, edge_norm


class SaintRWValSampler(BaseLoader):

    def __init__(self,
                 graph: Data,
                 output_indices: torch.LongTensor,
                 return_edge_index_type: str,
                 num_nodes: int,
                 walk_length: int,
                 sample_coverage: int = 0,
                 save_dir: Optional[str] = None,
                 **kwargs):

        self.walk_length = walk_length
        self.sample_coverage = sample_coverage

        self.N = num_nodes
        self.E = graph.num_edges

        self.adj = SparseTensor.from_edge_index(graph.edge_index, sparse_sizes=(graph.num_nodes, graph.num_nodes))
        self.adj = self.normalize_adjmat(self.adj, normalization='rw')
        self.edge_weight = self.adj.storage.value()
        self.adj.set_value_(torch.arange(self.E))

        self.original_graph = graph
        self.output_indices = output_indices

        assert return_edge_index_type in ['adj', 'edge_index']
        self.return_edge_index_type = return_edge_index_type

        super().__init__(self, **kwargs)

        if self.sample_coverage > 0:
            path = osp.join(save_dir or '', self.__filename__)
            assert save_dir is not None and osp.exists(path)
            self.node_norm, self.edge_norm = torch.load(path)
        else:
            self.node_norm = torch.ones(self.N)
            self.edge_norm = torch.ones(self.E)

    @property
    def __filename__(self):
        return (f'{self.N}_{self.walk_length}_'
                f'{self.sample_coverage}.pt')

    def __len__(self):
        return len(self.output_indices)

    @property
    def loader_len(self):
        return ceil(len(self.output_indices) / self.batch_size)

    def __getitem__(self, idx):
        return self.output_indices[idx]

    def __collate__(self, data_list):
        prime_nodes = torch.tensor(data_list, dtype=torch.long)
        node_idx = self.adj.random_walk(prime_nodes.flatten(), self.walk_length).view(-1).unique()
        adj, _ = self.adj.saint_subgraph(node_idx)

        _, _, edge_idx = adj.coo()
        if self.return_edge_index_type == 'adj':
            if self.sample_coverage > 0:
                adj.set_value_(self.edge_norm[edge_idx] * self.edge_weight[edge_idx], layout='csr')
            else:
                adj.set_value_(self.edge_weight[edge_idx], layout='csr')
        else:
            adj = torch.vstack([adj.storage.row(), adj.storage.col()])

        node_norm = self.node_norm[prime_nodes]
        node_norm /= node_norm.sum()
        graph = Data(x=self.original_graph.x[node_idx],
                     y=self.original_graph.y[node_idx],
                     edge_index=adj,
                     output_node_mask=torch.from_numpy(np.in1d(node_idx.numpy(), prime_nodes.numpy())),
                     node_norm=node_norm)

        return graph
