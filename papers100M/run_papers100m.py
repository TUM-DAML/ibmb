import sys
import os
sys.path.append(os.getcwd())
import logging
import pickle
import resource
import time
import traceback
import os.path as osp

import numpy as np
import seml
import torch
from sacred import Experiment

from dataloaders.get_loaders import get_loaders
from dataloaders.IBMBReadyLoader import IBMBReadyLoader
from data.data_preparation import check_consistence, load_data
from models.GCN import GCN
from train.trainer import Trainer

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
def run(mode,
        micro_batch,
        batch_order,
        batch_size,

        inference,
        LBMB_val,

        n_sampling_params=None,
        ladies_params=None,

        epoch_min=300,
        epoch_max=800,
        patience=100,
        lr=1e-3,
        graphmodel='gcn',
        num_layers=3,
        hidden_channels=256,
        reg=1.e-4,
        seed=None):
    try:

        check_consistence(mode, batch_order)
        logging.info(f'dataset: papers100M, graphmodel: {graphmodel}, mode: {mode}')

        graph, (train_indices, val_indices, test_indices) = load_data('papers100M', small_trainingset=1,
                                                                      pretransform=None)
        logging.info("Graph loaded!\n")

        graph.y = torch.nan_to_num(graph.y, nan=-1).reshape(-1).to(torch.long)
        graph.edge_index = None
        graph.adj_t = torch.load('/nfs/students/qian/adj.pt')
        logging.info("Graph processed!\n")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = Trainer(mode,
                          256,
                          micro_batch=micro_batch,
                          batch_size=batch_size,
                          epoch_max=epoch_max,
                          epoch_min=epoch_min,
                          patience=patience)

        comment = '_'.join(['papers100M', graphmodel, mode, ])

        if mode not in ['part', 'ppr']:
            (train_loader,
             self_val_loader,
             _,
             _,
             self_test_loader,
             _,
             _) = get_loaders(
                graph,
                (train_indices, val_indices, test_indices),
                batch_size,
                mode,
                batch_order,
                None,
                None,
                None,
                None,
                None,
                ladies_params,
                n_sampling_params,
                inference,
                False)
            if LBMB_val:
                with open('./papers100m_val_part_batches.pkl', 'rb') as handle:
                    batch_val_batches = pickle.load(handle)
                with open('./papers100m_val_ppr_batches.pkl', 'rb') as handle:
                    ppr_val_batches = pickle.load(handle)
                if inference:
                    with open('./papers100m_test_part_batches.pkl', 'rb') as handle:
                        batch_test_batches = pickle.load(handle)
                    with open('./papers100m_test_ppr_batches.pkl', 'rb') as handle:
                        ppr_test_batches = pickle.load(handle)
        else:
            if mode == 'part':
                with open('./papers100m_train_part_batches.pkl', 'rb') as handle:
                    train_batches = pickle.load(handle)
                with open('./papers100m_val_part_batches.pkl', 'rb') as handle:
                    val_batches = pickle.load(handle)
                    batch_val_batches = None
                with open('./papers100m_val_ppr_batches.pkl', 'rb') as handle:
                    ppr_val_batches = pickle.load(handle)
                if inference:
                    with open('./papers100m_test_part_batches.pkl', 'rb') as handle:
                        test_batches = pickle.load(handle)
                        batch_test_batches = None
                    with open('./papers100m_test_ppr_batches.pkl', 'rb') as handle:
                        ppr_test_batches = pickle.load(handle)
            elif mode == 'ppr':
                with open('./papers100m_train_ppr_batches.pkl', 'rb') as handle:
                    train_batches = pickle.load(handle)
                with open('./papers100m_val_ppr_batches.pkl', 'rb') as handle:
                    val_batches = pickle.load(handle)
                    ppr_val_batches = None
                with open('./papers100m_val_part_batches.pkl', 'rb') as handle:
                    batch_val_batches = pickle.load(handle)
                if inference:
                    with open('./papers100m_test_ppr_batches.pkl', 'rb') as handle:
                        test_batches = pickle.load(handle)
                        ppr_test_batches = None
                    with open('./papers100m_test_part_batches.pkl', 'rb') as handle:
                        batch_test_batches = pickle.load(handle)
            else:
                raise ValueError

            train_loader = IBMBReadyLoader(graph,
                                           batch_order,
                                           'adj',
                                           train_batches,
                                           graph.adj_t,
                                           batch_size=batch_size,
                                           shuffle=False)
            self_val_loader = IBMBReadyLoader(graph,
                                              batch_order,
                                              'adj',
                                              val_batches,
                                              graph.adj_t,
                                              batch_size=batch_size,
                                              shuffle=False)
            if inference:
                self_test_loader = IBMBReadyLoader(graph,
                                                   batch_order,
                                                   'adj',
                                                   test_batches,
                                                   graph.adj_t,
                                                   batch_size=batch_size,
                                                   shuffle=False)
            else:
                self_test_loader = None

        batch_val_loader = IBMBReadyLoader(graph,
                                           batch_order,
                                           'adj',
                                           batch_val_batches,
                                           graph.adj_t,
                                           batch_size=batch_size,
                                           shuffle=False) if batch_val_batches is not None else None
        batch_test_loader = IBMBReadyLoader(graph,
                                            batch_order,
                                           'adj',
                                           batch_test_batches,
                                           graph.adj_t,
                                           batch_size=batch_size,
                                           shuffle=False) if batch_test_batches is not None else None
        ppr_val_loader = IBMBReadyLoader(graph,
                                         batch_order,
                                           'adj',
                                           ppr_val_batches,
                                           graph.adj_t,
                                           batch_size=batch_size,
                                           shuffle=False) if ppr_val_batches is not None else None
        ppr_test_loader = IBMBReadyLoader(graph,
                                          batch_order,
                                           'adj',
                                           ppr_test_batches,
                                           graph.adj_t,
                                           batch_size=batch_size,
                                           shuffle=False) if ppr_test_batches is not None else None

        stamp = ''.join(str(time.time()).split('.')) + str(seed)
        logging.info(f'model info: {comment}/model_{stamp}.pt')
        if graphmodel == 'gcn':
            model = GCN(num_node_features=graph.num_node_features,
                        num_classes=graph.y.max().item() + 1,
                        hidden_channels=hidden_channels,
                        num_layers=num_layers).to(device)
        else:
            raise NotImplementedError

        trainer.train(train_loader,
                      self_val_loader,
                      ppr_val_loader,
                      batch_val_loader,
                      model=model,
                      lr=lr,
                      reg=reg,
                      comment=comment,
                      run_no=stamp)

        logging.info(f'after train: {torch.cuda.memory_allocated()}')
        logging.info(f'after train: {torch.cuda.memory_reserved()}')

        train_loader = None  # clear cache

        gpu_memory = torch.cuda.max_memory_allocated()
        if inference:
            model_dir = osp.join('./saved_models', comment)
            assert osp.isdir(model_dir)
            model_path = osp.join(model_dir, f'model_{stamp}.pt')
            model.load_state_dict(torch.load(model_path))
            model.eval()

            trainer.inference(self_val_loader,
                              ppr_val_loader,
                              batch_val_loader,
                              self_test_loader,
                              ppr_test_loader,
                              batch_test_loader,
                              model, )

        runtime_train_lst = []
        runtime_self_val_lst = []
        runtime_part_val_lst = []
        runtime_ppr_val_lst = []
        for curves in trainer.database['training_curves']:
            runtime_train_lst += curves['per_train_time']
            runtime_self_val_lst += curves['per_self_val_time']
            runtime_part_val_lst += curves['per_part_val_time']
            runtime_ppr_val_lst += curves['per_ppr_val_time']

        results = {
            'runtime_train_perEpoch': sum(runtime_train_lst) / len(runtime_train_lst),
            'runtime_selfval_perEpoch': sum(runtime_self_val_lst) / len(runtime_self_val_lst),
            'runtime_partval_perEpoch': sum(runtime_part_val_lst) / len(runtime_part_val_lst),
            'runtime_pprval_perEpoch': sum(runtime_ppr_val_lst) / len(runtime_ppr_val_lst),
            'gpu_memory': gpu_memory,
            'max_memory': 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'curves': trainer.database['training_curves'],
            # ...
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
