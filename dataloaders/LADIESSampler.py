import numpy as np
from typing import Union, List
from scipy.sparse import csr_matrix
from dataloaders.BaseLoader import BaseLoader
from torch_geometric.data import Data
import torch
from torch_geometric.utils import is_undirected
from torch_sparse import SparseTensor
from math import ceil
from data.data_utils import MyGraph


def ladies_sampler(batch_nodes: np.ndarray,
                   samp_num_list: Union[np.ndarray, List],
                   mat: csr_matrix):
    previous_nodes = batch_nodes
    adjs = []
    node_indices = [batch_nodes]

    for num in samp_num_list:
        U = mat[previous_nodes, :]

        pi = np.sum(U.power(2), axis=0).A1
        nonzero_mask = np.where(pi > 0)[0]
        p = pi[nonzero_mask]
        p /= p.sum()

        s_num = min(len(nonzero_mask), num)
        after_nodes = np.random.choice(nonzero_mask, s_num, p=p, replace=False)
        after_nodes = np.union1d(after_nodes, batch_nodes)
        adj = U[:, after_nodes]
        adjs.append((adj.indptr.astype(np.int64), adj.indices.astype(np.int64), adj.data, adj.shape))
        node_indices.append(after_nodes)
        previous_nodes = after_nodes

    adjs.reverse()
    node_indices.reverse()

    return adjs, node_indices


class LADIESSampler(BaseLoader):

    def __init__(self, graph: Data,
                 output_indices: torch.LongTensor,
                 return_edge_index_type: str,
                 samp_num_list: Union[np.ndarray, List],
                 batch_size: int = 1,
                 **kwargs):
        assert is_undirected(graph.edge_index, num_nodes=graph.num_nodes), "Assume the graph to be undirected"
        self.output_indices = output_indices.numpy()
        assert return_edge_index_type == 'adj'
        self.return_edge_index_type = return_edge_index_type
        self.batch_size = batch_size
        self.samp_num_list = samp_num_list

        self.original_graph = graph  # need to cache the original graph
        self.adj = SparseTensor.from_edge_index(graph.edge_index, sparse_sizes=(graph.num_nodes, graph.num_nodes))
        self.adj = BaseLoader.normalize_adjmat(self.adj, normalization='sym')
        self.adj = self.adj.to_scipy('csr')

        super().__init__(self.output_indices, batch_size=batch_size, **kwargs)

    def __getitem__(self, idx):
        return self.output_indices[idx]

    def __len__(self):
        return len(self.output_indices)

    @property
    def loader_len(self):
        return ceil(len(self.output_indices) / self.batch_size)

    def __collate__(self, data_list):
        batch_nodes = np.array(data_list)
        adjs_storage, node_indices = ladies_sampler(batch_nodes, self.samp_num_list, self.adj)
        aux_indices = node_indices[0]
        assert np.all(batch_nodes == node_indices[-1])

        adjs = []
        for indptr, indices, data, size in adjs_storage:
            adjs.append(SparseTensor(rowptr=torch.from_numpy(indptr),
                                     col=torch.from_numpy(indices),
                                     value=torch.from_numpy(data),
                                     sparse_sizes=size))

        subg = MyGraph(x=self.original_graph.x[aux_indices],
                       y=self.original_graph.y[batch_nodes],   # cannot use np.in1d since the order might differ
                       edge_index=adjs,
                       output_node_mask=torch.ones(len(batch_nodes), dtype=torch.bool))
        return subg
