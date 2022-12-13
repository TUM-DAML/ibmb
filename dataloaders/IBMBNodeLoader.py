import logging
import queue as Q
from heapq import heappush, heappop, heapify
from math import ceil
from typing import Optional, List, Tuple

import numba
import numpy as np
import torch
from torch.utils.data import Sampler
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected
# from torch_geometric.transforms.gdc import get_calc_ppr
from torch_sparse import SparseTensor
from tqdm import tqdm

from dataloaders.utils import topk_ppr_matrix
from .BaseLoader import BaseLoader


def get_pairs(ppr_mat: csr_matrix) -> np.ndarray:
    """
    Get symmetric ppr pairs. (Only upper triangle)

    :param ppr_mat:
    :return:
    """
    ppr_mat = ppr_mat + ppr_mat.transpose()

    ppr_mat = ppr_mat.tocoo()  # find issue: https://github.com/scipy/scipy/blob/v1.7.1/scipy/sparse/extract.py#L12-L40
    row, col, data = ppr_mat.row, ppr_mat.col, ppr_mat.data
    mask = (row > col)  # lu

    row, col, data = row[mask], col[mask], data[mask]
    sort_arg = np.argsort(data)[::-1]
    # sort_arg = parallel_sort.parallel_argsort(data)[::-1]

    # map prime_nodes to arange
    ppr_pairs = np.vstack((row[sort_arg], col[sort_arg])).T
    return ppr_pairs


@numba.njit(cache=True)
def prime_orient_merge(ppr_pairs: np.ndarray, primes_per_batch: int, num_nodes: int):
    """

    :param ppr_pairs:
    :param primes_per_batch:
    :param num_nodes:
    :return:
    """
    # cannot use list for id_primes_list, updating node_id_list[id_primes_list[id2]] require id_primes_list to be array
    id_primes_list = list(np.arange(num_nodes, dtype=np.int32).reshape(-1, 1))
    node_id_list = np.arange(num_nodes, dtype=np.int32)
    placeholder = np.zeros(0, dtype=np.int32)
    # size_flag = [{a} for a in np.arange(num_nodes, dtype=np.int32)]

    for i, j in ppr_pairs:
        id1, id2 = node_id_list[i], node_id_list[j]
        if id1 > id2:
            id1, id2 = id2, id1

        # if not (id1 in size_flag[id2] or id2 in size_flag[id1])
        if id1 != id2 and len(id_primes_list[id1]) + len(id_primes_list[id2]) <= primes_per_batch:
            id_primes_list[id1] = np.concatenate((id_primes_list[id1], id_primes_list[id2]))
            node_id_list[id_primes_list[id2]] = id1
            # node_id_list[j] = id1
            id_primes_list[id2] = placeholder

    prime_lst = list()
    ids = np.unique(node_id_list)

    for _id in ids:
        prime_lst.append(list(id_primes_list[_id]))

    return list(prime_lst)


def prime_post_process(loader, merge_max_size):
    h = [(len(p), p,) for p in loader]
    heapify(h)

    while len(h) > 1:
        len1, p1 = heappop(h)
        len2, p2 = heappop(h)
        if len1 + len2 <= merge_max_size:
            heappush(h, (len1 + len2, p1 + p2))
        else:
            heappush(h, (len1, p1,))
            heappush(h, (len2, p2,))
            break

    new_batch = []

    while len(h):
        _, p = heappop(h)
        new_batch.append(p)

    return new_batch


@numba.njit(cache=True, locals={'p1': numba.int64,
                                'p2': numba.int64,
                                'p3': numba.int64,
                                'new_list': numba.int64[::1]})
def merge_lists(lst1, lst2):
    p1, p2, p3 = numba.int64(0), numba.int64(0), numba.int64(0)
    new_list = np.zeros(len(lst1) + len(lst2), dtype=np.int64)

    while p2 < len(lst2) and p1 < len(lst1):
        if lst2[p2] <= lst1[p1]:
            new_list[p3] = lst2[p2]
            p2 += 1

            if lst2[p2 - 1] == lst1[p1]:
                p1 += 1

        elif lst2[p2] > lst1[p1]:
            new_list[p3] = lst1[p1]
            p1 += 1
        p3 += 1

    if p2 == len(lst2) and p1 == len(lst1):
        return new_list[:p3]
    elif p1 == len(lst1):
        rest = lst2[p2:]
    else:
        rest = lst1[p1:]

    p3_ = p3 + len(rest)
    new_list[p3: p3_] = rest

    return new_list[:p3_]


@numba.njit(cache=True, locals={'node_id_list': numba.int64[::1],
                                'placeholder': numba.int64[::1],
                                'id1': numba.int64,
                                'id2': numba.int64})
def aux_orient_merge(ppr_pairs, prime_indices, id_second_list, merge_max_size):
    thresh = numba.int64(merge_max_size * 1.0005)
    num_nodes = len(prime_indices)
    node_id_list = np.arange(num_nodes, dtype=np.int64)

    id_prime_list = list(np.arange(num_nodes, dtype=np.int64).reshape(-1, 1))
    size_flag = [{a} for a in np.arange(num_nodes, dtype=np.int64)]

    placeholder = np.zeros(0, dtype=np.int64)

    for (n1, n2) in ppr_pairs:
        id1, id2 = node_id_list[n1], node_id_list[n2]
        id1, id2 = (id1, id2) if id1 < id2 else (id2, id1)

        if id1 != id2 and not (id2 in size_flag[id1]) and not (id1 in size_flag[id2]):

            batch_second1 = id_second_list[id1]
            batch_second2 = id_second_list[id2]

            if len(batch_second1) + len(batch_second2) <= thresh:
                new_batch_second = merge_lists(batch_second1, batch_second2)
                if len(new_batch_second) <= merge_max_size:
                    batch_prime1 = id_prime_list[id1]
                    batch_prime2 = id_prime_list[id2]

                    new_batch_prime = np.concatenate((batch_prime1, batch_prime2))

                    id_prime_list[id1] = new_batch_prime
                    id_second_list[id1] = new_batch_second
                    id_second_list[id2] = placeholder

                    id_prime_list[id2] = placeholder

                    node_id_list[batch_prime2] = id1
                    size_flag[id1].update(size_flag[id2])
                    size_flag[id2].clear()
                else:
                    size_flag[id1].add(id2)
                    size_flag[id2].add(id1)
            else:
                size_flag[id1].add(id2)
                size_flag[id2].add(id1)

    prime_second_lst = list()
    ids = np.unique(node_id_list)

    for _id in ids:
        prime_second_lst.append((prime_indices[id_prime_list[_id]],
                                 id_second_list[_id]))

    return list(prime_second_lst)


def aux_post_process(loader, merge_max_size):
    # merge the smallest clusters first
    que = Q.PriorityQueue()
    for p, n in loader:
        que.put((len(n), (list(p), list(n))))

    while que.qsize() > 1:
        len1, (p1, n1) = que.get()
        len2, (p2, n2) = que.get()
        n = merge_lists(np.array(n1), np.array(n2))

        if len(n) > merge_max_size:
            que.put((len1, (p1, n1)))
            que.put((len2, (p2, n2)))
            break

        else:
            que.put((len(n), (p1 + p2, list(n))))

    new_batch = []

    while not que.empty():
        _, (p, n) = que.get()
        new_batch.append((np.array(p), np.array(n)))

    return new_batch


class IBMBNodeLoader(BaseLoader):
    """
    Batch-wise IBMB dataloader from paper Influence-Based Mini-Batching for Graph Neural Networks
    """

    def __init__(self, graph: Data,
                 batch_order: str,
                 output_indices: torch.LongTensor,
                 return_edge_index_type: str,
                 num_auxiliary_node_per_output: int,
                 num_output_nodes_per_batch: Optional[int] = None,
                 num_auxiliary_nodes_per_batch: Optional[int] = None,
                 alpha: float = 0.2,
                 eps: float = 1.e-4,
                 sampler: Sampler = None,
                 **kwargs):

        self.subgraphs = []
        self.node_wise_out_aux_pairs = []

        self.original_graph = None
        self.adj = None

        assert is_undirected(graph.edge_index, num_nodes=graph.num_nodes), "Assume the graph to be undirected"
        self.cache_data = kwargs['batch_size'] == 1
        self._batchsize = kwargs['batch_size']
        self.output_indices = output_indices.numpy()
        assert return_edge_index_type in ['adj', 'edge_index']
        self.return_edge_index_type = return_edge_index_type
        self.num_auxiliary_node_per_output = num_auxiliary_node_per_output
        self.num_output_nodes_per_batch = num_output_nodes_per_batch
        self.num_auxiliary_nodes_per_batch = num_auxiliary_nodes_per_batch
        self.alpha = alpha
        self.eps = eps

        self.create_node_wise_loader(graph)

        if len(self.node_wise_out_aux_pairs) > 2:   # <= 2 order makes no sense
            ys = [graph.y[out].numpy() for out, _ in self.node_wise_out_aux_pairs]
            sampler = self.define_sampler(batch_order,
                                          ys,
                                          graph.y.max().item() + 1)

        if not self.cache_data:
            self.original_graph = graph  # need to cache the original graph

        super().__init__(self.subgraphs if self.cache_data else self.node_wise_out_aux_pairs, sampler=sampler, **kwargs)

    def create_node_wise_loader(self, graph: Data):
        logging.info("Start PPR calculation")
        ppr_matrix, neighbors = topk_ppr_matrix(graph.edge_index,
                                                graph.num_nodes,
                                                self.alpha,
                                                self.eps,
                                                self.output_indices, self.num_auxiliary_node_per_output)

        ppr_matrix = ppr_matrix[:, self.output_indices]

        logging.info("Getting PPR pairs")
        ppr_pairs = get_pairs(ppr_matrix)

        assert (self.num_output_nodes_per_batch is not None) ^ (self.num_auxiliary_nodes_per_batch is not None)
        if self.num_output_nodes_per_batch is not None:
            logging.info("Output node oriented merging")
            output_list = prime_orient_merge(ppr_pairs, self.num_output_nodes_per_batch, len(self.output_indices))
            output_list = prime_post_process(output_list, self.num_output_nodes_per_batch)
            node_wise_out_aux_pairs = []

            if isinstance(neighbors, list):
                neighbors = np.array(neighbors, dtype=object)

            _union = lambda inputs: np.unique(np.concatenate(inputs))
            for p in output_list:
                node_wise_out_aux_pairs.append((self.output_indices[p], _union(neighbors[p]).astype(np.int64)))
        else:
            logging.info("Auxiliary node oriented merging")
            prime_second_lst = aux_orient_merge(ppr_pairs,
                                                self.output_indices,
                                                list(neighbors),
                                                merge_max_size=self.num_auxiliary_nodes_per_batch)
            node_wise_out_aux_pairs = aux_post_process(prime_second_lst, self.num_auxiliary_nodes_per_batch)

        self.indices_complete_check(node_wise_out_aux_pairs, self.output_indices)
        self.node_wise_out_aux_pairs = node_wise_out_aux_pairs

        if self.return_edge_index_type == 'adj':
            adj = SparseTensor.from_edge_index(graph.edge_index, sparse_sizes=(graph.num_nodes, graph.num_nodes))
            adj = self.normalize_adjmat(adj, normalization='rw')
        else:
            adj = None

        if self.cache_data:
            self.prepare_cache(graph, node_wise_out_aux_pairs, adj)
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
        return self.subgraphs[idx] if self.cache_data else self.node_wise_out_aux_pairs[idx]

    def __len__(self):
        return len(self.node_wise_out_aux_pairs)

    @property
    def loader_len(self):
        return ceil(len(self.node_wise_out_aux_pairs) / self._batchsize)

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
