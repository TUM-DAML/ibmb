import os
from math import ceil
from typing import Dict, Tuple, Union, Optional

from torch import LongTensor
from torch_geometric.data import Data

from dataloaders.ClusterGCNLoader import ClusterGCNLoader
from dataloaders.GraphSAINTRWSampler import SaintRWTrainSampler, SaintRWValSampler
from dataloaders.IBMBBatchLoader import IBMBBatchLoader
from dataloaders.IBMBNodeLoader import IBMBNodeLoader
from dataloaders.IBMBRandLoader import IBMBRandLoader
from dataloaders.IBMBRandfixLoader import IBMBRandfixLoader
from dataloaders.ShaDowLoader import ShaDowLoader
from dataloaders.LADIESSampler import LADIESSampler
from dataloaders.NeighborSamplingLoader import NeighborSamplingLoader

Loader = Union[
    ClusterGCNLoader,
    SaintRWTrainSampler,
    SaintRWValSampler,
    IBMBBatchLoader,
    IBMBNodeLoader,
    IBMBRandfixLoader,
    ShaDowLoader,
    LADIESSampler,
    NeighborSamplingLoader
]
EDGE_INDEX_TYPE = 'adj'


def num_out_nodes_per_batch_normalization(num_out_nodes: int,
                                          num_out_per_batch: int):
    num_batches = ceil(num_out_nodes / num_out_per_batch)
    return ceil(num_out_nodes / num_batches)


def get_loaders(graph: Data,
                splits: Tuple[LongTensor, LongTensor, LongTensor],
                batch_size: int,
                mode: str,
                batch_order: str,
                ppr_params: Optional[Dict],
                batch_params: Optional[Dict],
                rw_sampling_params: Optional[Dict],
                shadow_ppr_params: Optional[Dict],
                rand_ppr_params: Optional[Dict],
                ladies_params: Optional[Dict],
                n_sampling_params: Optional[Dict],
                inference: bool = True,
                ibmb_val: bool = True) -> Tuple[
    Optional[Loader],
    Optional[Loader],
    Optional[Loader],
    Optional[Loader],
    Optional[Loader],
    Optional[Loader],
    Optional[Loader]
]:
    train_indices, val_indices, test_indices = splits

    train_loader = None
    self_val_loader = None
    ppr_val_loader = None
    batch_val_loader = None
    self_test_loader = None
    ppr_test_loader = None
    batch_test_loader = None
    if mode == 'ppr':
        train_loader = IBMBNodeLoader(graph,
                                      batch_order,
                                      train_indices,
                                      EDGE_INDEX_TYPE,
                                      ppr_params['neighbor_topk'],
                                      num_output_nodes_per_batch=num_out_nodes_per_batch_normalization(
                                          len(train_indices), ppr_params['primes_per_batch']),
                                      num_auxiliary_nodes_per_batch=None,
                                      alpha=ppr_params['alpha'],
                                      eps=ppr_params['eps'],
                                      batch_size=batch_size,
                                      shuffle=False)    # must be false, instead we define our own order!
        self_val_loader = IBMBNodeLoader(graph,
                                         batch_order,
                                         val_indices,
                                         EDGE_INDEX_TYPE,
                                         ppr_params['neighbor_topk'],
                                         num_output_nodes_per_batch=num_out_nodes_per_batch_normalization(
                                             len(val_indices), ppr_params['primes_per_batch'] * 2),
                                         num_auxiliary_nodes_per_batch=None,
                                         alpha=ppr_params['alpha'],
                                         eps=ppr_params['eps'],
                                         batch_size=batch_size,
                                         shuffle=False)
        if inference:
            self_test_loader = IBMBNodeLoader(graph,
                                              batch_order,
                                              test_indices,
                                              EDGE_INDEX_TYPE,
                                              ppr_params['neighbor_topk'],
                                              num_output_nodes_per_batch=num_out_nodes_per_batch_normalization(
                                                  len(test_indices), ppr_params['primes_per_batch'] * 2),
                                              num_auxiliary_nodes_per_batch=None,
                                              alpha=ppr_params['alpha'],
                                              eps=ppr_params['eps'],
                                              batch_size=batch_size,
                                              shuffle=False)
    elif mode == 'part':
        train_loader = IBMBBatchLoader(graph,
                                       batch_order,
                                       batch_params['num_batches'][0],
                                       train_indices,
                                       EDGE_INDEX_TYPE,
                                       batch_params['part_topk'][0],
                                       alpha=batch_params['alpha'],
                                       batch_size=batch_size,
                                       shuffle=False)
        self_val_loader = IBMBBatchLoader(graph,
                                          batch_order,
                                          batch_params['num_batches'][1],
                                          val_indices,
                                          EDGE_INDEX_TYPE,
                                          batch_params['part_topk'][1],
                                          alpha=batch_params['alpha'],
                                          batch_size=batch_size,
                                          shuffle=False)
        if inference:
            self_test_loader = IBMBBatchLoader(graph,
                                               batch_order,
                                               batch_params['num_batches'][2],
                                               test_indices,
                                               EDGE_INDEX_TYPE,
                                               batch_params['part_topk'][1],
                                               alpha=batch_params['alpha'],
                                               batch_size=batch_size,
                                               shuffle=False)
    elif mode == 'rw_sampling':
        dir_name = f'./saint_cache'
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        train_loader = SaintRWTrainSampler(graph,
                                           train_indices,
                                           EDGE_INDEX_TYPE,
                                           graph.num_nodes,
                                           rw_sampling_params['batch_size'][0],
                                           rw_sampling_params['walk_length'],
                                           rw_sampling_params['num_steps'],
                                           rw_sampling_params['sample_coverage'],
                                           save_dir=dir_name,
                                           shuffle=True)
        self_val_loader = SaintRWValSampler(graph,
                                            val_indices,
                                            EDGE_INDEX_TYPE,
                                            graph.num_nodes,
                                            rw_sampling_params['walk_length'],
                                            rw_sampling_params['sample_coverage'],
                                            save_dir=dir_name,
                                            batch_size=rw_sampling_params['batch_size'][1],
                                            shuffle=True)
        if inference:
            self_test_loader = SaintRWValSampler(graph,
                                                 test_indices,
                                                 EDGE_INDEX_TYPE,
                                                 graph.num_nodes,
                                                 rw_sampling_params['walk_length'],
                                                 rw_sampling_params['sample_coverage'],
                                                 save_dir=dir_name,
                                                 batch_size=rw_sampling_params['batch_size'][1],
                                                 shuffle=True)
    elif mode == 'clustergcn':
        train_loader = ClusterGCNLoader(graph,
                                        batch_params['num_batches'][0],
                                        train_indices,
                                        EDGE_INDEX_TYPE,
                                        batch_size=batch_size,
                                        shuffle=True)
        self_val_loader = ClusterGCNLoader(graph,
                                           batch_params['num_batches'][1],
                                           val_indices,
                                           EDGE_INDEX_TYPE,
                                           batch_size=batch_size,
                                           shuffle=True)
        if inference:
            self_test_loader = ClusterGCNLoader(graph,
                                                batch_params['num_batches'][2],
                                                test_indices,
                                                EDGE_INDEX_TYPE,
                                                batch_size=batch_size,
                                                shuffle=True)
    elif mode == 'ppr_shadow':
        train_loader = ShaDowLoader(graph,
                                    train_indices,
                                    EDGE_INDEX_TYPE,
                                    shadow_ppr_params['neighbor_topk'],
                                    shadow_ppr_params['alpha'],
                                    shadow_ppr_params['eps'],
                                    batch_size=num_out_nodes_per_batch_normalization(
                                        len(train_indices), shadow_ppr_params['primes_per_batch']),
                                    shuffle=True)
        self_val_loader = ShaDowLoader(graph,
                                       val_indices,
                                       EDGE_INDEX_TYPE,
                                       shadow_ppr_params['neighbor_topk'],
                                       shadow_ppr_params['alpha'],
                                       shadow_ppr_params['eps'],
                                       batch_size=num_out_nodes_per_batch_normalization(
                                           len(val_indices), shadow_ppr_params['primes_per_batch'] * 2),
                                       shuffle=True)
        if inference:
            self_test_loader = ShaDowLoader(graph,
                                            test_indices,
                                            EDGE_INDEX_TYPE,
                                            shadow_ppr_params['neighbor_topk'],
                                            shadow_ppr_params['alpha'],
                                            shadow_ppr_params['eps'],
                                            batch_size=num_out_nodes_per_batch_normalization(
                                                len(test_indices), shadow_ppr_params['primes_per_batch'] * 2),
                                            shuffle=True)
    elif mode == 'rand':
        train_loader = IBMBRandLoader(graph,
                                      train_indices,
                                      EDGE_INDEX_TYPE,
                                      rand_ppr_params['neighbor_topk'],
                                      rand_ppr_params['alpha'],
                                      rand_ppr_params['eps'],
                                      batch_size=num_out_nodes_per_batch_normalization(
                                          len(train_indices), rand_ppr_params['primes_per_batch']), shuffle=True)
        self_val_loader = IBMBRandLoader(graph,
                                         val_indices,
                                         EDGE_INDEX_TYPE,
                                         rand_ppr_params['neighbor_topk'],
                                         rand_ppr_params['alpha'],
                                         rand_ppr_params['eps'],
                                         batch_size=num_out_nodes_per_batch_normalization(
                                             len(val_indices), rand_ppr_params['primes_per_batch'] * 2), shuffle=True)
        if inference:
            self_test_loader = IBMBRandLoader(graph,
                                              test_indices,
                                              EDGE_INDEX_TYPE,
                                              rand_ppr_params['neighbor_topk'],
                                              rand_ppr_params['alpha'],
                                              rand_ppr_params['eps'],
                                              batch_size=num_out_nodes_per_batch_normalization(
                                                  len(test_indices), rand_ppr_params['primes_per_batch'] * 2),
                                              shuffle=True)
    elif mode == 'randfix':
        train_loader = IBMBRandfixLoader(graph,
                                         batch_order,
                                         train_indices,
                                         EDGE_INDEX_TYPE,
                                         num_out_nodes_per_batch_normalization(
                                             len(train_indices), ppr_params['primes_per_batch']),
                                         rand_ppr_params['neighbor_topk'],
                                         rand_ppr_params['alpha'],
                                         rand_ppr_params['eps'],
                                         batch_size=batch_size,
                                         shuffle=False)
        self_val_loader = IBMBRandfixLoader(graph,
                                            batch_order,
                                            val_indices,
                                            EDGE_INDEX_TYPE,
                                            num_out_nodes_per_batch_normalization(
                                                len(val_indices), ppr_params['primes_per_batch'] * 2),
                                            rand_ppr_params['neighbor_topk'],
                                            rand_ppr_params['alpha'],
                                            rand_ppr_params['eps'],
                                            batch_size=batch_size,
                                            shuffle=False)
        if inference:
            self_test_loader = IBMBRandfixLoader(graph,
                                                 batch_order,
                                                 test_indices,
                                                 EDGE_INDEX_TYPE,
                                                 num_out_nodes_per_batch_normalization(
                                                     len(test_indices), ppr_params['primes_per_batch'] * 2),
                                                 rand_ppr_params['neighbor_topk'],
                                                 rand_ppr_params['alpha'],
                                                 rand_ppr_params['eps'],
                                                 batch_size=batch_size,
                                                 shuffle=False)
    elif mode == 'ladies':
        train_loader = LADIESSampler(graph,
                                     train_indices,
                                     EDGE_INDEX_TYPE,
                                     [ladies_params['sample_size'][0]] * ladies_params['num_layers'],
                                     batch_size=ceil(len(train_indices) / ladies_params['num_batches'][0]),
                                     shuffle=True)
        self_val_loader = LADIESSampler(graph,
                                        val_indices,
                                        EDGE_INDEX_TYPE,
                                        [ladies_params['sample_size'][1]] * ladies_params['num_layers'],
                                        batch_size=ceil(len(val_indices) / ladies_params['num_batches'][1]),
                                        shuffle=True)
        if inference:
            self_test_loader = LADIESSampler(graph,
                                             test_indices,
                                             EDGE_INDEX_TYPE,
                                             [ladies_params['sample_size'][2]] * ladies_params['num_layers'],
                                             batch_size=ceil(len(test_indices) / ladies_params['num_batches'][2]),
                                             shuffle=True)
    elif mode == 'n_sampling':
        train_loader = NeighborSamplingLoader(graph,
                                              sizes=n_sampling_params['n_nodes'],
                                              node_idx=train_indices,
                                              batch_size=ceil(len(train_indices) / n_sampling_params['num_batches'][0]),
                                              shuffle=True)
        self_val_loader = NeighborSamplingLoader(graph,
                                                 sizes=n_sampling_params['n_nodes'],
                                                 node_idx=val_indices,
                                                 batch_size=ceil(
                                                     len(val_indices) / n_sampling_params['num_batches'][1]),
                                                 shuffle=True)
        if inference:
            self_test_loader = NeighborSamplingLoader(graph,
                                                      sizes=n_sampling_params['n_nodes'],
                                                      node_idx=test_indices,
                                                      batch_size=ceil(
                                                          len(test_indices) / n_sampling_params['num_batches'][2]),
                                                      shuffle=True)
    else:
        raise NotImplementedError

    if ibmb_val:
        if mode != 'ppr' and ppr_params is not None:
            ppr_val_loader = IBMBNodeLoader(graph,
                                            batch_order,
                                            val_indices,
                                            EDGE_INDEX_TYPE,
                                            ppr_params['neighbor_topk'],
                                            num_output_nodes_per_batch=num_out_nodes_per_batch_normalization(
                                                len(val_indices), ppr_params['primes_per_batch'] * 2),
                                            num_auxiliary_nodes_per_batch=None,
                                            alpha=ppr_params['alpha'],
                                            eps=ppr_params['eps'],
                                            batch_size=batch_size,
                                            shuffle=False)
            if inference:
                ppr_test_loader = IBMBNodeLoader(graph,
                                                 batch_order,
                                                 test_indices,
                                                 EDGE_INDEX_TYPE,
                                                 ppr_params['neighbor_topk'],
                                                 num_output_nodes_per_batch=num_out_nodes_per_batch_normalization(
                                                     len(test_indices), ppr_params['primes_per_batch'] * 2),
                                                 num_auxiliary_nodes_per_batch=None,
                                                 alpha=ppr_params['alpha'],
                                                 eps=ppr_params['eps'],
                                                 batch_size=batch_size,
                                                 shuffle=False)
        if mode != 'part' and batch_params is not None:
            batch_val_loader = IBMBBatchLoader(graph,
                                               batch_order,
                                               batch_params['num_batches'][1],
                                               val_indices,
                                               EDGE_INDEX_TYPE,
                                               batch_params['part_topk'][1],
                                               alpha=batch_params['alpha'],
                                               batch_size=batch_size,
                                               shuffle=False)
            if inference:
                batch_test_loader = IBMBBatchLoader(graph,
                                                    batch_order,
                                                    batch_params['num_batches'][2],
                                                    test_indices,
                                                    EDGE_INDEX_TYPE,
                                                    batch_params['part_topk'][1],
                                                    alpha=batch_params['alpha'],
                                                    batch_size=batch_size,
                                                    shuffle=False)

    return (train_loader,
            self_val_loader,
            ppr_val_loader,
            batch_val_loader,
            self_test_loader,
            ppr_test_loader,
            batch_test_loader)
