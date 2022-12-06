import logging
import pickle
import resource
import time
import traceback

import numpy as np
import seml
import torch
from sacred import Experiment

from batching import get_loader
from data.customed_dataset import MYDataset
from data.data_preparation import check_consistence, load_data, config_transform
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
def run(dataset_name,
        graphmodel,
        mode,
        neighbor_sampling,
        micro_batch,
        num_batches,
        batch_order,
        reg,
        hidden_channels,

        inference,
        LBMB_val,

        cache_sub_adj=True,
        cache_origin_adj=False,

        ppr_params=None,
        n_sampling_params=None,
        ladies_params=None,

        epoch_min=300,
        epoch_max=800,
        patience=100,
        lr=1e-3,
        num_layers=3,
        seed=None):
    try:

        check_consistence(mode, neighbor_sampling, batch_order['ordered'], batch_order['sampled'])
        logging.info(
            f'dataset: {dataset_name}, graphmodel: {graphmodel}, mode: {mode}, neighbor_sampling: {neighbor_sampling}')

        start_time = time.time()
        graph, (train_indices, val_indices, test_indices) = load_data(dataset_name, small_trainingset=1)
        logging.info("Graph loaded!\n")
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

        start_time = time.time()
        graph.y = torch.nan_to_num(graph.y, nan=-1).reshape(-1).to(torch.long)
        graph.edge_index = None
        graph.adj_t = torch.load('/nfs/students/qian/adj.pt')
        logging.info("Graph processed!\n")
        graph_preprocess_time = time.time() - start_time

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = Trainer(mode,
                          neighbor_sampling,
                          num_batches,
                          micro_batch=micro_batch,
                          batch_size=1,
                          epoch_max=epoch_max,
                          epoch_min=epoch_min,
                          patience=patience)

        comment = '_'.join([dataset_name,
                            graphmodel,
                            mode,
                            neighbor_sampling,
                            str(micro_batch),
                            str(merge_max_size[0])])

        # train & val
        start_time = time.time()

        train_loader = None
        val_loader = [None, None, None]
        test_loader = [None, None, None]
        if mode not in ['part', 'ppr']:
            train_loader, val_loader, test_loader = get_loader(mode,
                                                               dataset_name,
                                                               neighbor_sampling,
                                                               graph.adj_t,
                                                               (train_indices, val_indices, test_indices),
                                                               neighbor_topk,
                                                               ppr_params,
                                                               part_topk=[1, 1],
                                                               num_nodes=graph.num_nodes,
                                                               merge_max_sizes=merge_max_size,
                                                               num_batches=num_batches,
                                                               primes_per_batch=primes_per_batch,
                                                               num_layers=num_layers,
                                                               partition_diffusion_param=0.05,
                                                               n_sampling_params=n_sampling_params,
                                                               rw_sampling_params={},
                                                               LBMB_val=False,
                                                               inference=inference)
            if LBMB_val:
                with open('./papers100m_val_part_batches.pkl', 'rb') as handle:
                    val_loader[1] = pickle.load(handle)
                with open('./papers100m_val_ppr_batches.pkl', 'rb') as handle:
                    val_loader[2] = pickle.load(handle)
                if inference:
                    with open('./papers100m_test_part_batches.pkl', 'rb') as handle:
                        test_loader[1] = pickle.load(handle)
                    with open('./papers100m_test_ppr_batches.pkl', 'rb') as handle:
                        test_loader[2] = pickle.load(handle)
        elif mode == 'part':
            with open('./papers100m_train_part_batches.pkl', 'rb') as handle:
                train_loader = pickle.load(handle)
            with open('./papers100m_val_part_batches.pkl', 'rb') as handle:
                val_loader[0] = pickle.load(handle)
            if inference:
                with open('./papers100m_test_part_batches.pkl', 'rb') as handle:
                    test_loader[0] = pickle.load(handle)
            if LBMB_val:
                with open('./papers100m_val_ppr_batches.pkl', 'rb') as handle:
                    val_loader[2] = pickle.load(handle)
                if inference:
                    with open('./papers100m_test_ppr_batches.pkl', 'rb') as handle:
                        test_loader[2] = pickle.load(handle)
        elif mode == 'ppr':
            with open('./papers100m_train_ppr_batches.pkl', 'rb') as handle:
                train_loader = pickle.load(handle)
            with open('./papers100m_val_ppr_batches.pkl', 'rb') as handle:
                val_loader[0] = pickle.load(handle)
            if inference:
                with open('./papers100m_test_ppr_batches.pkl', 'rb') as handle:
                    test_loader[0] = pickle.load(handle)
            if LBMB_val:
                with open('./papers100m_val_part_batches.pkl', 'rb') as handle:
                    val_loader[1] = pickle.load(handle)
                if inference:
                    with open('./papers100m_test_part_batches.pkl', 'rb') as handle:
                        test_loader[1] = pickle.load(handle)

        train_prep_time = time.time() - start_time

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

        stamp = ''.join(str(time.time()).split('.')) + str(seed)
        logging.info(f'model info: {comment}/model_{stamp}.pt')
        if graphmodel == 'gcn':
            model = GCN(num_node_features=graph.num_node_features,
                        num_classes=graph.y.max().item() + 1,
                        hidden_channels=hidden_channels,
                        num_layers=num_layers).to(device)
        else:
            raise NotImplementedError

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

        gpu_memory = torch.cuda.max_memory_allocated()
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
                              clear_cache=True,)

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
            'disk_loading_time': disk_loading_time,
            'graph_preprocess_time': graph_preprocess_time,
            'train_prep_time': train_prep_time,
            'caching_time': caching_time,
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
