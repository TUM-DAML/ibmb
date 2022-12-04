import os
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch_geometric.data import NeighborSampler
from torch_sparse import SparseTensor

from .MySaintSampler import SaintRWTrainSampler, SaintRWValSampler
from .layerwise_sampling import LadiesLoader
from .loader_clustergcn import ClusterGCN_loader
from .loader_part_based import partition_fixed_loader
from .loader_part_ppr_mix import partition_ppr_loader
from .loader_prime_orient_ppr import ppr_fixed_loader as prime_ppr_loader
from .loader_rand import RandLoader, rand_fixed_loader
from .loader_shadow import RandShadowLoader
from data.data_preparation import get_partitions
from neighboring.ppr_power_iteration import ppr_power_iter


def get_loader(mode: str,
               dataset_name: str,
               neighbor_sampling: str,
               adj: SparseTensor,
               prime_indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
               neighbor_topk: List[int],
               ppr_params: Dict,
               part_topk: List[float],
               num_nodes: int,
               merge_max_sizes: List[int],
               num_batches: List[int],
               primes_per_batch: List[int],
               num_layers: int,
               partition_diffusion_param: float,
               n_sampling_params: dict,
               rw_sampling_params: dict,
               LBMB_val: bool = True,
               train: bool = True,
               val: bool = True,
               inference: bool = False):

    assert train or val or inference

    train_indices, val_indices, test_indices = prime_indices
    train_num_batch, val_num_batch, test_num_batch = num_batches
    train_neighbor_topk, val_neighbor_topk, test_neighbor_topk = neighbor_topk
    train_part_topk, val_part_topk = part_topk
    test_part_topk = val_part_topk
    train_merge_max_size, val_merge_max_size, test_merge_max_size = merge_max_sizes
    train_primes_per_batch, val_primes_per_batch, test_primes_per_batch = primes_per_batch

    # partition based
    train_partitions, val_partitions, test_partitions = None, None, None
    if train:
        train_partitions = get_partitions(mode, adj, train_indices, train_num_batch)

    if val:
        if val_num_batch == train_num_batch and train_partitions is not None and mode != 'part+ppr':
            # lazy
            val_partitions = train_partitions
        else:
            val_partitions = get_partitions(mode, adj, val_indices, val_num_batch, force=LBMB_val)

    if inference:
        if test_num_batch == train_num_batch and train_partitions is not None and mode != 'part+ppr':
            test_partitions = train_partitions
        elif test_num_batch == val_num_batch and val_partitions is not None and mode != 'part+ppr':
            test_partitions = val_partitions
        else:
            test_partitions = get_partitions(mode, adj, test_indices, test_num_batch)

    # ppr based
    train_mat = None
    val_mat = None
    train_neighbors = None
    val_neighbors = None
    test_mat = None
    test_neighbors = None
    if 'ppr' in [mode, neighbor_sampling] or LBMB_val:
        neighbors, pprmat = ppr_power_iter(adj,
                                           dataset_name,
                                           topk=train_neighbor_topk,
                                           alpha=partition_diffusion_param,
                                           thresh=ppr_params['pushflowthresh'])
        if train:
            train_mat = pprmat[train_indices, :][:, train_indices]
            train_neighbors = list(neighbors[train_indices])

        if val:
            val_mat = pprmat[val_indices, :][:, val_indices]
            val_neighbors = list(neighbors[val_indices])

        if inference:
            test_mat = pprmat[test_indices, :][:, test_indices]
            test_neighbors = list(neighbors[test_indices])

    train_loader = None
    val_loader = [None, None, None]
    test_loader = [None, None, None]

    if mode == 'clustergcn':
        if train:
            train_loader = ClusterGCN_loader(train_partitions, train_indices)
        if val:
            val_loader[0] = ClusterGCN_loader(val_partitions, val_indices)
        if inference:
            test_loader[0] = ClusterGCN_loader(test_partitions, test_indices)

    elif mode == 'part':
        if neighbor_sampling == 'ladies':
            adj_scipy = adj.to_scipy('csr')

            if train:
                raw = ClusterGCN_loader(train_partitions, train_indices)
                prime_batchlist = [p for (p, _) in raw]
                train_loader = LadiesLoader(prime_batchlist, np.full(num_layers, train_merge_max_size), adj_scipy)

            if val:
                raw = ClusterGCN_loader(val_partitions, val_indices)
                prime_batchlist = [p for (p, _) in raw]
                val_loader[0] = LadiesLoader(prime_batchlist, np.full(num_layers, val_merge_max_size), adj_scipy)

            if inference:
                raw = ClusterGCN_loader(test_partitions, test_indices)
                prime_batchlist = [p for (p, _) in raw]
                test_loader[0] = LadiesLoader(prime_batchlist, np.full(num_layers, test_merge_max_size), adj_scipy)

        elif neighbor_sampling in ['batch_ppr', 'batch_hk']:
            if train:
                train_loader = partition_fixed_loader(neighbor_sampling,
                                                      adj,
                                                      num_nodes,
                                                      train_partitions,
                                                      train_indices,
                                                      topk=int(train_merge_max_size * train_part_topk),
                                                      partition_diffusion_param=partition_diffusion_param)

            if val:
                val_loader[0] = partition_fixed_loader(neighbor_sampling,
                                                       adj,
                                                       num_nodes,
                                                       val_partitions,
                                                       val_indices,
                                                       topk=int(val_merge_max_size * val_part_topk),
                                                       partition_diffusion_param=partition_diffusion_param)

            if inference:
                test_loader[0] = partition_fixed_loader(neighbor_sampling,
                                                        adj,
                                                        num_nodes,
                                                        test_partitions,
                                                        test_indices,
                                                        topk=int(test_merge_max_size * test_part_topk),
                                                        partition_diffusion_param=partition_diffusion_param)

    elif mode == 'ppr':
        ## sedondary-oriented batching
        # loader = aux_ppr_loader(ppr_mat,
        #                          prime_indices,
        #                          neighbors,
        #                          merge_max_size=merge_max_size)

        # primary-oriented batching
        if train:
            train_loader = prime_ppr_loader(train_mat,
                                            train_indices,
                                            train_neighbors,
                                            train_primes_per_batch)

        if val:
            val_loader[0] = prime_ppr_loader(val_mat,
                                             val_indices,
                                             val_neighbors,
                                             val_primes_per_batch)

        if inference:
            test_loader[0] = prime_ppr_loader(test_mat,
                                              test_indices,
                                              test_neighbors,
                                              test_primes_per_batch)

    elif mode == 'part+ppr':
        if train:
            train_loader = partition_ppr_loader(train_partitions, train_indices, np.array(train_neighbors, dtype=object))
        if val:
            val_loader[0] = partition_ppr_loader(val_partitions, val_indices, np.array(val_neighbors, dtype=object))
        if inference:
            test_loader[0] = partition_ppr_loader(test_partitions, test_indices, np.array(test_neighbors, dtype=object))

    elif mode == 'ppr_shadow':
        if train:
            train_loader = RandShadowLoader(train_indices, train_neighbors, train_primes_per_batch)
        if val:
            val_loader[0] = RandShadowLoader(val_indices, val_neighbors, val_primes_per_batch)
        if inference:
            test_loader[0] = RandShadowLoader(test_indices, test_neighbors, test_primes_per_batch)

    elif mode == 'n_sampling':
        train_n_batches, val_n_batches, test_n_batches = n_sampling_params['num_batches']

        if train:
            train_batch_size = len(train_indices) // train_n_batches + ((len(train_indices) % train_n_batches) > 0)
            train_loader = NeighborSampler(adj, node_idx=torch.from_numpy(train_indices),
                                           sizes=n_sampling_params['n_nodes'],
                                           batch_size=train_batch_size,
                                           shuffle=True, num_workers=0)

        if val:
            val_batch_size = len(val_indices) // val_n_batches + ((len(val_indices) % val_n_batches) > 0)
            val_loader[0] = NeighborSampler(adj, node_idx=torch.from_numpy(val_indices),
                                            sizes=n_sampling_params['n_nodes'],
                                            batch_size=val_batch_size,
                                            shuffle=True, num_workers=0)

        if inference:
            test_batch_size = len(test_indices) // test_n_batches + ((len(test_indices) % test_n_batches) > 0)
            test_loader[0] = NeighborSampler(adj, node_idx=torch.from_numpy(test_indices),
                                             sizes=n_sampling_params['n_nodes'],
                                             batch_size=test_batch_size,
                                             shuffle=True, num_workers=0)

    elif mode == 'rw_sampling':
        dir_name = f'./saint_cache'
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        if train:
            train_loader = SaintRWTrainSampler(adj, num_nodes,
                                               batch_size=rw_sampling_params['batch_size'][0],
                                               walk_length=rw_sampling_params['walk_length'],
                                               num_steps=rw_sampling_params['num_steps'],
                                               sample_coverage=rw_sampling_params['sample_coverage'],
                                               save_dir=dir_name)

        if val:
            val_loader[0] = SaintRWValSampler(adj, val_indices, num_nodes,
                                              walk_length=rw_sampling_params['walk_length'],
                                              sample_coverage=rw_sampling_params['sample_coverage'],
                                              save_dir=dir_name,
                                              batch_size=rw_sampling_params['batch_size'][1])

        if inference:
            test_loader[0] = SaintRWValSampler(adj, test_indices, num_nodes,
                                               walk_length=rw_sampling_params['walk_length'],
                                               sample_coverage=rw_sampling_params['sample_coverage'],
                                               save_dir=dir_name,
                                               batch_size=rw_sampling_params['batch_size'][1])

    elif mode == 'rand':
        if train:
            train_loader = RandLoader(train_indices, train_neighbors, train_merge_max_size)
        if val:
            val_loader[0] = RandLoader(val_indices, val_neighbors, val_merge_max_size)
        if inference:
            test_loader[0] = RandLoader(test_indices, test_neighbors, test_merge_max_size)
    elif mode == 'randfix':
        if train:
            train_loader = rand_fixed_loader(train_indices, train_neighbors, train_merge_max_size)
        if val:
            val_loader[0] = rand_fixed_loader(val_indices, val_neighbors, val_merge_max_size)
        if inference:
            test_loader[0] = rand_fixed_loader(test_indices, test_neighbors, test_merge_max_size)

    elif mode == 'ladies':
        if train:
            train_loader = LadiesLoader(train_indices,
                                        np.full(num_layers, train_merge_max_size),
                                        adj.to_scipy('csr'),
                                        train_num_batch)
        if val:
            val_loader[0] = LadiesLoader(val_indices,
                                         np.full(num_layers, val_merge_max_size),
                                         adj.to_scipy('csr'),
                                         val_num_batch)
        if inference:
            test_loader[0] = LadiesLoader(test_indices,
                                          np.full(num_layers, test_merge_max_size),
                                          adj.to_scipy('csr'),
                                          test_num_batch)
    else:
        raise NotImplementedError

    if LBMB_val and val:
        if val_partitions is not None and not mode == 'part':
            val_loader[1] = partition_fixed_loader('batch_ppr',
                                                   adj,
                                                   num_nodes,
                                                   val_partitions,
                                                   val_indices,
                                                   topk=int((num_nodes // val_num_batch + 1) * val_part_topk),
                                                   partition_diffusion_param=partition_diffusion_param)

        if val_mat is not None and val_neighbors is not None and not mode == 'ppr':
            val_loader[2] = prime_ppr_loader(val_mat,
                                             val_indices,
                                             val_neighbors,
                                             val_primes_per_batch)

    if LBMB_val and inference:
        if test_partitions is not None and not mode == 'part':
            test_loader[1] = partition_fixed_loader('batch_ppr',
                                                    adj,
                                                    num_nodes,
                                                    test_partitions,
                                                    test_indices,
                                                    topk=int((num_nodes // test_num_batch + 1) * test_part_topk),
                                                    partition_diffusion_param=partition_diffusion_param)

        if test_mat is not None and test_neighbors is not None and not mode == 'ppr':
            test_loader[2] = prime_ppr_loader(test_mat,
                                              test_indices,
                                              test_neighbors,
                                              test_primes_per_batch)

    return train_loader, val_loader, test_loader
