import logging
import resource
import time
import traceback

import os.path as osp
import numpy as np
import seml
import torch
from sacred import Experiment

from dataloaders.get_loaders import get_loaders
from data.data_preparation import check_consistence, load_data, GraphPreprocess
from models.get_model import get_model
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
def run(dataset_name,
        mode,
        batch_size,
        micro_batch,
        batch_order,
        inference,
        LBMB_val,
        small_trainingset,

        ppr_params,
        batch_params,
        n_sampling_params=None,
        rw_sampling_params=None,
        ladies_params=None,
        shadow_ppr_params=None,
        rand_ppr_params=None,

        graphmodel='gcn',
        hidden_channels=256,
        reg=0.,
        num_layers=3,
        heads=None,

        epoch_min=300,
        epoch_max=800,
        patience=100,
        lr=1e-3,

        seed=None, ):
    try:

        check_consistence(mode, batch_order)
        logging.info(f'dataset: {dataset_name}, graphmodel: {graphmodel}, mode: {mode}')

        graph, (train_indices, val_indices, test_indices) = load_data(dataset_name,
                                                                      small_trainingset,
                                                                      GraphPreprocess(True, True))
        logging.info("Graph loaded!\n")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = Trainer(mode,
                          batch_params['num_batches'][0],
                          micro_batch=micro_batch,
                          batch_size=batch_size,
                          epoch_max=epoch_max,
                          epoch_min=epoch_min,
                          patience=patience)

        comment = '_'.join([dataset_name,
                            graphmodel,
                            mode])

        (train_loader,
         self_val_loader,
         ppr_val_loader,
         batch_val_loader,
         self_test_loader,
         ppr_test_loader,
         batch_test_loader) = get_loaders(
            graph,
            (train_indices, val_indices, test_indices),
            batch_size,
            mode,
            batch_order,
            ppr_params,
            batch_params,
            rw_sampling_params,
            shadow_ppr_params,
            rand_ppr_params,
            ladies_params,
            n_sampling_params,
            inference,
            LBMB_val)

        stamp = ''.join(str(time.time()).split('.')) + str(seed)

        logging.info(f'model info: {comment}/model_{stamp}.pt')
        model = get_model(graphmodel,
                          graph.num_node_features,
                          graph.y.max().item() + 1,
                          hidden_channels,
                          num_layers,
                          heads,
                          device)

        trainer.train(train_loader,
                      self_val_loader,
                      ppr_val_loader,
                      batch_val_loader,
                      model=model,
                      lr=lr,
                      reg=reg,
                      comment=comment,
                      run_no=stamp)

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

            trainer.full_graph_inference(model, graph, val_indices, test_indices)

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
