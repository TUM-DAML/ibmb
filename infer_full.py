import numpy as np
import torch
import logging
import resource
import time
from sacred import Experiment
import seml
import os
from data.data_preparation import load_data, graph_preprocess
from data.customed_dataset import MYDataset
from models import DeeperGCN, GAT, SAGEModel
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
        graphmodel,
        num_batches,
        hidden_channels,
        num_layers,
        epoch_min=300,
        epoch_max=800,
        patience=100,
        lr=1e-3):
    

    logging.info(
        f'dataset: {dataset_name}, graphmodel: {graphmodel}, '
        f'num_batches: {num_batches}')

    device = 'cuda' if torch.cuda.is_available else 'cpu'

    # common preprocess
    start_time = time.time()
    graph, (_, val_indices, test_indices) = load_data(dataset_name, 1)
    logging.info("Graph loaded!\n")
    disk_loading_time = time.time() - start_time

    start_time = time.time()
    graph_preprocess(graph)
    logging.info("Graph processed!\n")
    graph_preprocess_time = time.time() - start_time

    trainer = Trainer('part',
                      'batch_ppr',
                      num_batches, 
                      micro_batch=1,
                      batch_size=1,
                      epoch_max=epoch_max,
                      epoch_min=epoch_min,
                      patience=patience)

    # train & val
    start_time = time.time()

    val_prep_time = time.time() - start_time

    # inference
    start_time = time.time()

    infer_prep_time = time.time() - start_time

    # common preprocess
    start_time = time.time()
    dataset = MYDataset(graph.x.cpu().detach().numpy(),
                        graph.y.cpu().detach().numpy(),
                        graph.adj_t.to_scipy('csr'),
                        train_loader=None,
                        val_loader=[None, None],
                        test_loader=[None, None],
                        batch_order={'ordered': False, 'sampled': False},
                        cache_sub_adj=True,
                        cache_origin_adj=False,
                        cache=True)
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
                    heads=4).to(device)
    elif graphmodel == 'sage':
        model = SAGEModel(num_node_features=graph.num_node_features,
                          num_classes=graph.y.max().item() + 1,
                          hidden_channels=hidden_channels,
                          num_layers=num_layers).to(device)

    for _file in os.listdir(f'../pretrained/{graphmodel}_{dataset_name}/'):
        no = _file.split('.')[0].split('_')[1]
        trainer.inference(dataset=dataset,
                          model=model,
                          val_nodes=val_indices,
                          test_nodes=test_indices,
                          adj=graph.adj_t,
                          x=graph.x,
                          y=graph.y,
                          file_dir='../pretrained',
                          comment=f'{graphmodel}_{dataset_name}',
                          run_no=no, 
                          full_infer=True, 
                          record_numbatch=False)
        

    results = {
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
