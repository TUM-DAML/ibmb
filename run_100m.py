import numpy as np
import torch
import logging
import time
from sacred import Experiment
import pickle
import seml
import os
from data.data_preparation import check_consistence, load_data, graph_preprocess, get_partitions, get_ppr_mat, config_transform
from data.customed_dataset import MYDataset
from neighboring import get_neighbors
from neighboring.pernode_ppr_neighbor import topk_ppr_matrix
from batching import get_loader
from models import DeeperGCN
from train.trainer import Trainer
from models.DeeperGCN import MyGCNConv


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
        neighbor_sampling,
        diffusion_param,
        small_trainingset,
        batch_size,
        micro_batch,
        num_batches,
        batch_order, 
        part_topk,
        inference,
        
        cache_sub_adj = True,
        cache_origin_adj = False,
        
        ppr_params = None, 
        n_sampling_params = None,
        rw_sampling_params = None,
        ladies_params = None,

        epoch_min=300,
        epoch_max=800,
        patience=100,
        lr=1e-3,
        reg = 1e-4,
        hidden_channels=256,
        num_layers=3, 
        device = 'cuda'):
    
    check_consistence(mode, neighbor_sampling, batch_order['ordered'], batch_order['sampled'])
    logging.info(f'dataset: {dataset_name}, mode: {mode}, neighbor_sampling: {neighbor_sampling}')

    start_time = time.time()
    graph, (train_indices, val_indices, test_indices) = load_data(dataset_name, small_trainingset)
    logging.info("Graph loaded!\n")
    disk_loading_time = time.time() - start_time

    merge_max_size, neighbor_topk, primes_per_batch, N_sampling_params = config_transform((len(train_indices), 
                                                                                           len(val_indices), 
                                                                                           len(test_indices)), 
                                                                                          mode, neighbor_sampling, graph.num_nodes, 
                                                                                          num_batches,
                                                                                          ppr_params, ladies_params, n_sampling_params)

    start_time = time.time()
    graph.y = torch.nan_to_num(graph.y, nan=-1).to(torch.long).reshape(-1)
    graph.edge_index = None
    graph.adj_t = torch.load('/nfs/students/qian/data/papers100m/adj.pt')
    # graph_preprocess(graph)
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

    comment = '_'.join([dataset_name,
                        mode,
                        neighbor_sampling,
                        str(small_trainingset),
                        str(batch_size),
                        str(micro_batch),
                        str(merge_max_size[0]),
                        str(part_topk)])

    # train & val
    start_time = time.time()
    
    if mode == 'part':
#         with open('/nfs/students/qian/train_part.pkl', 'rb') as handle:
#             train_loader = pickle.load(handle)
#         logging.info("Train loader!\n")

#         with open('/nfs/students/qian/val_part.pkl', 'rb') as handle:
#             val_loader = pickle.load(handle)
#         val_loader = [val_loader, None]
        raise ValueError
        
    elif mode == 'ppr':
        scipy_adj = graph.adj_t.to_scipy('csr')
        train_mat, train_neighbors = topk_ppr_matrix(scipy_adj, 
                                                     diffusion_param, 
                                                     ppr_params['pushflowthresh'], 
                                                     train_indices, 
                                                     topk=ppr_params['neighbor_topk'], 
                                                     normalization='sym')
        train_mat = train_mat[:, train_indices]
        val_mat, val_neighbors = topk_ppr_matrix(scipy_adj, 
                                                 diffusion_param, 
                                                 ppr_params['pushflowthresh'], 
                                                 val_indices, 
                                                 topk=ppr_params['neighbor_topk'], 
                                                 normalization='sym')
        val_mat = val_mat[:, val_indices]
        
    else:
        train_mat = None
        val_mat = None
        train_neighbors = None
        val_neighbors = None
        
    train_loader = get_loader(mode,
                          neighbor_sampling,
                          graph.adj_t,
                          graph.num_nodes,
                          merge_max_size[0],
                          num_batches[0],
                          primes_per_batch[0], 
                          num_layers,
                          diffusion_param,
                          N_sampling_params,
                          rw_sampling_params,
                          train=True,
                          partitions=None,
                          part_topk=part_topk[0],
                          prime_indices=train_indices,
                          neighbors=train_neighbors,
                          ppr_mat=train_mat)

    val_loader = get_loader(mode,
                        neighbor_sampling,
                        graph.adj_t,
                        graph.num_nodes,
                        merge_max_size[1],
                        num_batches[1],
                        primes_per_batch[1], 
                        num_layers,
                        diffusion_param,
                        N_sampling_params,
                        rw_sampling_params,
                        train=False,
                        partitions=None,
                        part_topk=part_topk[1],
                        prime_indices=val_indices,
                        neighbors=val_neighbors,
                        ppr_mat=val_mat)
        
    logging.info("Val loader!\n")
    train_prep_time = time.time() - start_time
    
    with open('/nfs/students/qian/papers100m_val_LBMB.pkl', 'rb') as handle:
        val_loader[1] = pickle.load(handle)

    # inference
    start_time = time.time()
    if inference:
        if mode == 'part':
            # with open('/nfs/students/qian/test_part.pkl', 'rb') as handle:
            #     test_loader = pickle.load(handle)
            # test_loader = [test_loader, None]
            raise ValueError
        elif mode == 'ppr':
            test_mat, test_neighbors = topk_ppr_matrix(scipy_adj, 
                                                       diffusion_param, 
                                                       ppr_params['pushflowthresh'], 
                                                       test_indices, 
                                                       topk=ppr_params['neighbor_topk'], 
                                                       normalization='sym')
            test_mat = test_mat[:, test_indices]
        else:
            test_mat = None
            test_neighbors = None
            
        test_loader = get_loader(mode,
                             neighbor_sampling,
                             graph.adj_t,
                             graph.num_nodes,
                             merge_max_size[2],
                             num_batches[2],
                             primes_per_batch[2], 
                             num_layers,
                             diffusion_param,
                             N_sampling_params,
                             rw_sampling_params,
                             train=False,
                             partitions=None,
                             part_topk=part_topk[1],
                             prime_indices=test_indices,
                             neighbors=test_neighbors,
                             ppr_mat=test_mat)
        
        with open('/nfs/students/qian/papers100m_test_LBMB.pkl', 'rb') as handle:
            test_loader[1] = pickle.load(handle)
        logging.info("Test loader!\n")
        
    else:
        test_loader = [None, None]

    infer_prep_time = time.time() - start_time
    
#     return {'disk_loading_time': disk_loading_time, 
#             'graph_preprocess_time': graph_preprocess_time,
#             'train_prep_time': train_prep_time, 
#             'infer_prep_time': infer_prep_time,}

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
                        cache = not (mode in ['n_sampling', 'rand', 'rw_sampling'] or 'ladies' in [mode, neighbor_sampling]))
    caching_time = time.time() - start_time

    stamp = ''.join(str(time.time()).split('.'))
    logging.info(f'model info: {comment}/model_{stamp}.pt')
    
    model = DeeperGCN(num_node_features=graph.num_node_features,
                      num_classes=graph.y.max().item() + 1,
                      hidden_channels=hidden_channels,
                      num_layers=num_layers).to(device)

    trainer.train(dataset=dataset,
                  model=model,
                  lr=lr,
                  reg=reg,
                  train_nodes=train_indices,
                  val_nodes=val_indices,
                  comment=comment,
                  run_no=stamp)

    logging.info(f'after train: {torch.cuda.memory_allocated()}')
    logging.info(f'after train: {torch.cuda.memory_reserved()}')

    dataset.set_split('train')
    dataset.clear_cur_cache()
    
    if inference:
        trainer.inference(dataset=dataset,
                          model=model,
                          val_nodes=val_indices,
                          test_nodes=test_indices,
                          adj=graph.adj_t,
                          x=graph.x,
                          y=graph.y,
                          comment=comment,
                          run_no=stamp,
                          full_infer=False, 
                          clear_cache=True)   # don't do inference with a reference of x

    runtime_train_lst = []
    runtime_self_val_lst = []
    runtime_LBMB_val_lst = []
    for curves in trainer.database['training_curves']:
        runtime_train_lst += curves['per_train_time']
        runtime_self_val_lst += curves['per_self_val_time']
        runtime_LBMB_val_lst += curves['per_LBMB_val_time']

    results = {
        'disk_loading_time': disk_loading_time,
        'graph_preprocess_time': graph_preprocess_time,
        'train_prep_time': train_prep_time,
        'infer_prep_time': infer_prep_time,
        'caching_time': caching_time,
        'runtime_train_perEpoch': sum(runtime_train_lst) / len(runtime_train_lst),
        'runtime_selfval_perEpoch': sum(runtime_self_val_lst) / len(runtime_self_val_lst),
        'runtime_LBMBval_perEpoch': sum(runtime_LBMB_val_lst) / len(runtime_LBMB_val_lst),
        'gpu_memory': torch.cuda.max_memory_allocated(),
        'curves': trainer.database['training_curves'],
        # ...
    }

    for key, item in trainer.database.items():
        if key != 'training_curves':
            results[f'{key}_record'] = item
            item = np.array(item)
            results[f'{key}_stats'] = (item.mean(), item.std(),) if len(item) else (0., 0.,)

    return results
