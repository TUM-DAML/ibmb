from typing import List, Union, Tuple
import logging

import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_sparse import SparseTensor

from dataloaders.MySampler import OrderedSampler, ConsecutiveSampler
from data.data_utils import get_pair_wise_distance
from data.modified_tsp import tsp_heuristic


class BaseLoader(torch.utils.data.DataLoader):
    """
    Batch-wise IBMB dataloader from paper Influence-Based Mini-Batching for Graph Neural Networks
    """

    def __init__(self,
                 dataset,
                 *args,
                 **kwargs):
        super().__init__(dataset, collate_fn=self.__collate__, **kwargs)

    def __getitem__(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def loader_len(self):
        raise NotImplementedError

    def __collate__(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def normalize_adjmat(cls, adj: SparseTensor, normalization: str):
        """
        Normalize SparseTensor adjacency matrix.

        :param adj:
        :param normalization:
        :return:
        """

        assert normalization in ['sym', 'rw'], f"Unsupported normalization type {normalization}"
        assert isinstance(adj, SparseTensor), f"Expect SparseTensor type, got {type(adj)}"

        adj = adj.fill_value(1.)
        degree = adj.sum(0)

        degree[degree == 0.] = 1e-12
        deg_inv = 1 / degree

        if normalization == 'sym':
            deg_inv_sqrt = deg_inv ** 0.5
            adj = adj * deg_inv_sqrt.reshape(1, -1)
            adj = adj * deg_inv_sqrt.reshape(-1, 1)
        elif normalization == 'rw':
            adj = adj * deg_inv.reshape(-1, 1)

        return adj

    @classmethod
    def indices_complete_check(cls,
                               loader: List[Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]],
                               output_indices: Union[torch.Tensor, np.ndarray]):
        if isinstance(output_indices, torch.Tensor):
            output_indices = output_indices.cpu().numpy()

        outs = []
        for out, aux in loader:
            if isinstance(out, torch.Tensor):
                out = out.cpu().numpy()
            if isinstance(aux, torch.Tensor):
                aux = aux.cpu().numpy()

            assert np.all(np.in1d(out, aux)), "Not all output nodes are in aux nodes!"
            outs.append(out)

        outs = np.sort(np.concatenate(outs))
        assert np.all(outs == np.sort(output_indices)), "Output nodes missing or duplicate!"

    @classmethod
    def get_subgraph(cls,
                     out_indices: torch.Tensor,
                     graph: Data,
                     return_edge_index_type: str,
                     adj: SparseTensor,
                     **kwargs):
        if return_edge_index_type == 'adj':
            assert adj is not None

        if return_edge_index_type == 'adj':
            subg = Data(x=graph.x[out_indices],
                        y=graph.y[out_indices],
                        edge_index=adj[out_indices, :][:, out_indices])
        elif return_edge_index_type == 'edge_index':
            edge_index, edge_attr = subgraph(out_indices,
                                             graph.edge_index,
                                             graph.edge_attr,
                                             relabel_nodes=True,
                                             num_nodes=graph.num_nodes,
                                             return_edge_mask=False)
            subg = Data(x=graph.x[out_indices],
                        y=graph.y[out_indices],
                        edge_index=edge_index,
                        edge_attr=edge_attr)
        else:
            raise NotImplementedError

        for k, v in kwargs.items():
            subg[k] = v

        return subg

    @classmethod
    def define_sampler(cls,
                       batch_order: str,
                       ys: List[Union[torch.Tensor, np.ndarray, List]],
                       num_classes: int,
                       dist_type: str = 'kl'):
        if batch_order == 'rand':
            logging.info("Running with random order")
            sampler = RandomSampler(ys)
        elif batch_order in ['order', 'sample']:
            kl_div = get_pair_wise_distance(ys, num_classes, dist_type=dist_type)
            if batch_order == 'order':
                best_perm, _ = tsp_heuristic(kl_div)
                logging.info(f"Running with given order: {best_perm}")
                sampler = OrderedSampler(best_perm)
            else:
                logging.info("Running with weighted sampling")
                sampler = ConsecutiveSampler(kl_div)
        else:
            raise ValueError

        return sampler
