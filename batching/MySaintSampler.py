import copy
import os.path as osp
from typing import Optional
import torch
import numpy as np
from tqdm import tqdm
from torch_sparse import SparseTensor


class SaintRWTrainSampler(torch.utils.data.DataLoader):    
    
    def __init__(self, adj_t: SparseTensor, num_nodes: int, batch_size: int, walk_length: int, num_steps: int = 1,
                 sample_coverage: int = 0, save_dir: Optional[str] = None,
                 log: bool = True, **kwargs):

        self.walk_length = walk_length
        self.num_steps = num_steps
        self.__batch_size__ = batch_size
        self.sample_coverage = sample_coverage
        self.log = log

        self.N = num_nodes
        self.E = len(adj_t.storage.value())

        self.adj = SparseTensor(
            row=adj_t.storage._row, col=adj_t.storage._col,
            value=torch.arange(self.E),
            sparse_sizes=adj_t.sizes())
        
        self.edge_weight = adj_t.storage.value()

        super(SaintRWTrainSampler,
              self).__init__(self, batch_size=1, collate_fn=self.__collate__,
                             **kwargs)

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
        start = torch.randint(0, self.N, (batch_size, ), dtype=torch.long)
        node_idx = self.adj.random_walk(start.flatten(), self.walk_length)
        return node_idx.view(-1)
    

    def __getitem__(self, idx):
        node_idx = self.__sample_nodes__(self.__batch_size__).unique()
        adj, _ = self.adj.saint_subgraph(node_idx)
        return node_idx, adj
    

    def __collate__(self, data_list):
        assert len(data_list) == 1
        node_idx, adj = data_list[0]

        _, _, edge_idx = adj.coo()

        if self.sample_coverage > 0:
            node_norm = self.node_norm[node_idx]
            adj.set_value_(self.edge_norm[edge_idx] * self.edge_weight[edge_idx], layout='csr')
        else:
            node_norm = None
            adj.set_value_(self.edge_weight[edge_idx], layout='csr')

        return (None, node_idx), adj, node_norm
    
    
    def __compute_norm__(self):
        node_count = torch.zeros(self.N, dtype=torch.float)
        edge_count = torch.zeros(self.E, dtype=torch.float)

        loader = torch.utils.data.DataLoader(self, batch_size=200,
                                             collate_fn=lambda x: x,
                                             num_workers=self.num_workers)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.N * self.sample_coverage)
            pbar.set_description('Compute GraphSAINT normalization')

        num_samples = total_sampled_nodes = 0
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


class SaintRWValSampler(torch.utils.data.DataLoader):
    
    def __init__(self, adj_t: SparseTensor, prime_nodes: np.ndarray, num_nodes: int, 
                 walk_length: int, 
                 sample_coverage: int = 0, save_dir: Optional[str] = None, **kwargs):

        self.walk_length = walk_length
        self.sample_coverage = sample_coverage

        self.N = num_nodes
        self.E = len(adj_t.storage.value())

        self.adj = SparseTensor(
            row=adj_t.storage._row, col=adj_t.storage._col,
            value=torch.arange(self.E),
            sparse_sizes=adj_t.sizes())
        
        self.edge_weight = adj_t.storage.value()
        
        self.prime_nodes = prime_nodes

        super(SaintRWValSampler,
              self).__init__(self, collate_fn=self.__collate__,
                             **kwargs)

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
        return len(self.prime_nodes)
    
    
    def loader_len(self):
        return len(self.prime_nodes) // self.batch_size + (len(self.prime_nodes) % self.batch_size > 0)
    

    def __getitem__(self, idx):
        return self.prime_nodes[idx]
    

    def __collate__(self, data_list):
        prime_nodes = torch.tensor(data_list, dtype=torch.long)
        node_idx = self.adj.random_walk(prime_nodes.flatten(), self.walk_length).view(-1).unique()
        adj, _ = self.adj.saint_subgraph(node_idx)

        _, _, edge_idx = adj.coo()

        if self.sample_coverage > 0:
            adj.set_value_(self.edge_norm[edge_idx] * self.edge_weight[edge_idx], layout='csr')
        else:
            adj.set_value_(self.edge_weight[edge_idx], layout='csr')

        return (prime_nodes, node_idx), adj, self.node_norm[prime_nodes]