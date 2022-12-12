import logging
import os
import resource
import traceback

import numpy as np
import seml
import torch
from sacred import Experiment

from dataloaders.get_loaders import get_loaders
from data.data_preparation import load_data, GraphPreprocess
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
        full_graph_chunks,
        model_dir,

        ppr_params=None,
        batch_params=None,
        n_sampling_params=None,
        rw_sampling_params=None,
        ladies_params=None,
        shadow_ppr_params=None,
        rand_ppr_params=None,

        graphmodel='gcn',
        hidden_channels=256,
        num_layers=3,
        heads=None, ):
    try:

        logging.info(f'dataset: {dataset_name}, graphmodel: {graphmodel}, mode: {mode}')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        graph, (train_indices, val_indices, test_indices) = load_data(dataset_name, 1,
                                                                      GraphPreprocess(True, True))
        logging.info("Graph loaded!\n")

        trainer = Trainer(mode, full_graph_chunks, batch_size=1, )

        (_,
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
            'rand',
            ppr_params,
            batch_params,
            rw_sampling_params,
            shadow_ppr_params,
            rand_ppr_params,
            ladies_params,
            n_sampling_params,
            inference=True,
            ibmb_val=False)

        model = get_model(graphmodel,
                          graph.num_node_features,
                          graph.y.max().item() + 1,
                          hidden_channels,
                          num_layers,
                          heads,
                          device)

        for _file in os.listdir(model_dir):
            if not _file.endswith('.pt'):
                continue
            model_path = os.path.join(model_dir, _file)
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

        results = {
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
