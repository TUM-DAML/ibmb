import logging
import resource
import time

import psutil
import seml
import torch
from sacred import Experiment

from batching import get_loader
from data.customed_dataset import MYDataset
from data.data_preparation import check_consistence, load_data, graph_preprocess, config_transform
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
        mode,
        neighbor_sampling,
        diffusion_param,
        small_trainingset,
        batch_size,
        micro_batch,
        num_batches,
        batch_order,
        part_topk,
        reg,
        hidden_channels,

        cache_sub_adj=True,
        cache_origin_adj=False,

        ppr_params=None,
        n_sampling_params=None,
        rw_sampling_params=None,
        ladies_params=None,

        epoch_min=1,
        epoch_max=2,
        patience=100,
        lr=1e-3,
        num_layers=3,
        heads=None,
        device='cuda'):
    check_consistence(mode, neighbor_sampling, batch_order['ordered'], batch_order['sampled'])
    logging.info(
        f'dataset: {dataset_name}, graphmodel: {graphmodel}, mode: {mode}, neighbor_sampling: {neighbor_sampling}')

    start_time = time.time()
    logging.info(f'mem begin: {psutil.Process().memory_info().rss}\n')
    graph, (train_indices, val_indices, test_indices) = load_data(dataset_name, small_trainingset)
    disk_loading_time = time.time() - start_time

    merge_max_size, neighbor_topk, primes_per_batch, ppr_params = config_transform(dataset_name,
                                                                                   graphmodel,
                                                                                   (len(train_indices),
                                                                                    len(val_indices),
                                                                                    len(test_indices)),
                                                                                   mode, neighbor_sampling,
                                                                                   graph.num_nodes,
                                                                                   num_batches,
                                                                                   ppr_params, ladies_params, )

    logging.info(f'mem graph: {psutil.Process().memory_info().rss}\n')

    start_time = time.time()
    graph_preprocess(graph)
    graph_preprocess_time = time.time() - start_time
    logging.info(f'mem process adj: {psutil.Process().memory_info().rss}\n')

    trainer = Trainer(mode,
                      neighbor_sampling,
                      num_batches,
                      micro_batch=micro_batch,
                      batch_size=batch_size,
                      epoch_max=epoch_max,
                      epoch_min=epoch_min,
                      patience=patience)

    comment = '_'.join([dataset_name,
                        graphmodel,
                        mode,
                        neighbor_sampling,
                        str(small_trainingset),
                        str(batch_size),
                        str(micro_batch),
                        str(merge_max_size[0]),
                        str(part_topk[0]), ])

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
                                                       inference=True)

    train_prep_time = time.time() - start_time

    # inference
    start_time = time.time()
    infer_prep_time = time.time() - start_time

    # not recording infer memory
    test_loader = [None, None, None]

    logging.info(f'mem del testloader: {psutil.Process().memory_info().rss}\n')

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

    logging.info(f'mem trainer cache: {psutil.Process().memory_info().rss}\n')

    stamp = ''.join(str(time.time()).split('.'))
    logging.info(f'model info: {comment}/model_{stamp}.pt')
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

    logging.info(f'mem model: {psutil.Process().memory_info().rss}\n')

    del graph
    logging.info(f'mem del graph: {psutil.Process().memory_info().rss}\n')

    if len(dataset.train_loader) > 1:
        trainer.train(dataset=dataset,
                      model=model,
                      lr=lr,
                      reg=reg,
                      train_nodes=train_indices,
                      val_nodes=val_indices,
                      comment=comment,
                      run_no=stamp)
    else:
        raise NotImplementedError

    avg_memory = psutil.Process().memory_info().rss

    logging.info(f'mem after train: {psutil.Process().memory_info().rss}\n')

    results = {
        'disk_loading_time': disk_loading_time,
        'graph_preprocess_time': graph_preprocess_time,
        'train_prep_time': train_prep_time,
        'infer_prep_time': infer_prep_time,
        'caching_time': caching_time,
        'gpu_memory': torch.cuda.max_memory_allocated(),
        'max_memory': 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + 1024 * resource.getrusage(
            resource.RUSAGE_CHILDREN).ru_maxrss,
        'avg_memory': avg_memory,
        # ...
    }

    return results
