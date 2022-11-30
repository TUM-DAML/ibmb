import numpy as np
import torch
import logging
import resource
import time
from sacred import Experiment
import pickle
import seml
import traceback
import os
from numba.typed import List
from data.data_preparation import check_consistence, load_data, graph_preprocess, get_partitions, get_ppr_mat, \
    config_transform
from data.customed_dataset import MYDataset
from neighboring import get_neighbors
from neighboring.ppr_power_iteration import ppr_power_iter
from batching import get_loader
from models import DeeperGCN, GAT, SAGEModel
from train.trainer import Trainer
from copy import deepcopy

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(dataset_name,
        graphmodel,
        mode,
        neighbor_sampling,
        diffusion_param,
        num_batches,
        hidden_channels,
        part_topk,

        micro_batch=1,
        batch_size=1,
        small_trainingset=1,
        batch_order={'ordered': False, 'sampled': False},

        cache_sub_adj=True,
        cache_origin_adj=False,

        inference=True,

        ppr_params=None,
        n_sampling_params=None,
        rw_sampling_params=None,
        ladies_params=None,

        epoch_min=300,
        epoch_max=800,
        patience=100,
        num_layers=3,
        heads=None,):
    try:
        seed = np.random.choice(2 ** 16)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        check_consistence(mode, neighbor_sampling, batch_order['ordered'], batch_order['sampled'])

        logging.info(
            f'dataset: {dataset_name}, graphmodel: {graphmodel}, mode: {mode}, neighbor_sampling: {neighbor_sampling}, '
            f'num_batches: {num_batches}, batch_order: {batch_order}, part_topk: {part_topk}, micro_batch: {micro_batch}, '
            f'rw_sampling_params: {rw_sampling_params}, n_sampling_params: {n_sampling_params}, ladies_params: {ladies_params}')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        start_time = time.time()
        graph, (train_indices, val_indices, test_indices) = load_data(dataset_name, small_trainingset)
        logging.info("Graph loaded!\n")
        disk_loading_time = time.time() - start_time

        merge_max_size, neighbor_topk, primes_per_batch = config_transform(dataset_name,
                                                                           graphmodel,
                                                                           (len(train_indices),
                                                                            len(val_indices),
                                                                            len(test_indices)),
                                                                           mode, neighbor_sampling,
                                                                           graph.num_nodes,
                                                                           num_batches,
                                                                           ppr_params, ladies_params, )

        start_time = time.time()
        graph_preprocess(graph)
        logging.info("Graph processed!\n")
        graph_preprocess_time = time.time() - start_time

        trainer = Trainer(mode,
                          neighbor_sampling,
                          num_batches,
                          micro_batch=micro_batch,
                          batch_size=batch_size,
                          epoch_max=epoch_max,
                          epoch_min=epoch_min,
                          patience=patience)

        # train & val
        start_time = time.time()

        train_loader, val_loader, test_loader = get_loader(mode,
                                                           dataset_name,
                                                           neighbor_sampling,
                                                           graph.adj_t,
                                                           (train_indices, val_indices, test_indices),
                                                           neighbor_topk,
                                                           ppr_params,
                                                           part_topk,
                                                           graph.num_nodes,
                                                           merge_max_size,
                                                           num_batches,
                                                           primes_per_batch,
                                                           num_layers,
                                                           diffusion_param,
                                                           n_sampling_params,
                                                           rw_sampling_params,
                                                           LBMB_val=False,
                                                           train=False,
                                                           val=True,
                                                           inference=inference,)

        val_prep_time = time.time() - start_time

        # inference
        start_time = time.time()

        infer_prep_time = time.time() - start_time

        # common preprocess
        start_time = time.time()
        dataset = MYDataset(graph.x.cpu().detach().numpy(),
                            graph.y.cpu().detach().numpy(),
                            graph.adj_t.to_scipy('csr'),
                            train_loader=train_loader,
                            val_loader=val_loader,
                            test_loader=test_loader,
                            batch_order=batch_order,
                            cache_sub_adj=cache_sub_adj,
                            cache_origin_adj=cache_origin_adj,
                            cache=not (mode in ['n_sampling', 'rand', 'rw_sampling', 'ppr_shadow'] or
                                       'ladies' in [mode, neighbor_sampling]))
        caching_time = time.time() - start_time

        #     return dataset
        if graphmodel == 'gcn':
            model = DeeperGCN(num_node_features=graph.num_node_features,
                              num_classes=graph.y.max().item() + 1,
                              hidden_channels=hidden_channels,
                              num_layers=num_layers).to(device)

        elif graphmodel == 'gat':
            model = GAT(in_channels=graph.num_node_features,
                        hidden_channels=hidden_channels,
                        out_channels=graph.y.max().item() + 1,
                        num_layers=num_layers,
                        heads=heads).to(device)
        elif graphmodel == 'sage':
            model = SAGEModel(num_node_features=graph.num_node_features,
                              num_classes=graph.y.max().item() + 1,
                              hidden_channels=hidden_channels,
                              num_layers=num_layers).to(device)
        else:
            raise NotImplementedError

        for _file in os.listdir(f'./pretrained/{graphmodel}_{dataset_name}/'):
            no = _file.split('.')[0].split('_')[1]
            trainer.inference(dataset=dataset,
                              model=model,
                              val_nodes=val_indices,
                              test_nodes=test_indices,
                              adj=graph.adj_t,
                              x=graph.x,
                              y=graph.y,
                              file_dir='./pretrained',
                              comment=f'{graphmodel}_{dataset_name}',
                              run_no=no,
                              full_infer=False,
                              record_numbatch=True)
        #         break

        results = {
            'seed': seed,
            'disk_loading_time': disk_loading_time,
            'graph_preprocess_time': graph_preprocess_time,
            'val_prep_time': val_prep_time,
            'infer_prep_time': infer_prep_time,
            'caching_time': caching_time,
            'gpu_memory': torch.cuda.max_memory_allocated(),
            'max_memory': 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        }

        for key, item in trainer.database.items():
            if key != 'training_curves':
                results[f'{key}_record'] = item
                item = np.array(item)
                results[f'{key}_stats'] = (item.mean(), item.std(),) if len(item) else (0., 0.,)

        return results
    except:
        traceback.print_exc()
        exit()
