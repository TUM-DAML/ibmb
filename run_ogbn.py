import numpy as np
import torch
import logging
import resource
import time
from sacred import Experiment
import traceback
import seml
from data.data_preparation import check_consistence, load_data, graph_preprocess, config_transform
from data.customed_dataset import MYDataset
from batching import get_loader
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

        inference,
        LBMB_val,

        cache_sub_adj=True,
        cache_origin_adj=False,

        ppr_params=None,
        # mixed_ppr_params = None,  # only for partition + ppr variant
        n_sampling_params=None,
        rw_sampling_params=None,
        ladies_params=None,

        epoch_min=300,
        epoch_max=800,
        patience=100,
        lr=1e-3,
        num_layers=3,
        heads=None,
        seed=None):
    try:
        seed = np.random.choice(2 ** 16)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        check_consistence(mode, neighbor_sampling, batch_order['ordered'], batch_order['sampled'])
        logging.info(
            f'dataset: {dataset_name}, graphmodel: {graphmodel}, mode: {mode}, neighbor_sampling: {neighbor_sampling}')

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

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
                                                           LBMB_val,
                                                           inference)

        train_prep_time = time.time() - start_time

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

        stamp = ''.join(str(time.time()).split('.')) + str(seed)
        logging.info(f'model info: {comment}/model_{stamp}.pt')
        model = None
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
            trainer.train_single_batch(dataset=dataset,
                                       model=model,
                                       lr=lr,
                                       reg=reg,
                                       val_per_epoch=5,
                                       comment=comment,
                                       run_no=stamp)

        logging.info(f'after train: {torch.cuda.memory_allocated()}')
        logging.info(f'after train: {torch.cuda.memory_reserved()}')

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
                              run_no=stamp)

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
            'seed': seed,
            'disk_loading_time': disk_loading_time,
            'graph_preprocess_time': graph_preprocess_time,
            'train_prep_time': train_prep_time,
            'infer_prep_time': infer_prep_time,
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
