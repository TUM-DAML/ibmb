import sys
import os
sys.path.append(os.getcwd())
import logging
import time
from collections import defaultdict

import numpy as np
import seml
from sacred import Experiment
from sklearn.metrics import f1_score

from data.data_preparation import load_data
from models.GCN import GCN, MyGCNConv
from models.chunk_beta import *

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
def run(model_path='',
        num_chunks=64,
        hidden_channels=256,
        num_layers=3,
        device='cuda'):
    start_time = time.time()
    graph, (train_indices, val_indices, test_indices) = load_data('papers100M', 1, None)
    logging.info("Graph loaded!\n")
    disk_loading_time = time.time() - start_time

    start_time = time.time()
    graph.y = torch.nan_to_num(graph.y, nan=-1).to(torch.long).reshape(-1)
    graph.edge_index = None
    graph.adj_t = torch.load('/nfs/students/qian/adj.pt')
    logging.info("Graph processed!\n")
    graph_preprocess_time = time.time() - start_time

    model = GCN(num_node_features=graph.num_node_features,
                num_classes=graph.y.max().item() + 1,
                hidden_channels=hidden_channels,
                num_layers=num_layers).to(device)

    # full infer
    model.load_state_dict(torch.load(model_path))

    model.eval()
    start_time = time.time()
    mask = np.union1d(val_indices, test_indices)
    val_mask = np.in1d(mask, val_indices)
    test_mask = np.in1d(mask, test_indices)
    assert np.all(np.invert(val_mask) == test_mask)
    adj = chunk_adj_row(graph.adj_t, num_chunks)
    x = graph.x
    y = graph.y
    del graph
    mask = torch.zeros(x.shape[0], dtype=torch.bool)
    mask[val_indices] = True
    mask[test_indices] = True
    idx = get_chunk_idx(len(mask), num_chunks)

    with torch.no_grad():
        for i, l in enumerate(model.layers):
            if isinstance(l, MyGCNConv):
                if i == len(model.layers) - 5 and mask is not None:
                    for j, (s, e) in enumerate(idx):
                        adj[j] = adj[j][mask[s:e], :]
                        if adj[j].nnz() == 0:
                            adj[j] = None
                x = general_chunk_forward_beta(l.lin, x, num_chunks)
                x = chunked_sp_matmul_beta(adj, x, num_chunks, reduce=l.aggr, device=l.weight.device)
            elif isinstance(l, (torch.nn.Linear, torch.nn.LayerNorm)):
                x = general_chunk_forward_beta(l, x, num_chunks)
            else:  # relu, dropout
                x = chunk_nonparam_layer(x, l, num_chunks)

    x = x.numpy()

    database = defaultdict(list)

    for cat in ['val', 'test']:
        nodes = val_indices if cat == 'val' else test_indices
        _mask = val_mask if cat == 'val' else test_mask
        pred = np.argmax(x[_mask], axis=1)
        true = y.detach().numpy()[nodes]

        acc = (pred == true).sum() / len(true)
        f1 = f1_score(true, pred, average='macro', zero_division=0)

        database[f'full_{cat}_accs'].append(acc)
        database[f'full_{cat}_f1s'].append(f1)

        logging.info("full_{}_acc: {:.3f}, full_{}_f1: {:.3f}, ".format(cat, acc, cat, f1))

    database['full_inference_time'].append(time.time() - start_time)

    database['disk_loading_time'].append(disk_loading_time)
    database['graph_preprocess_time'].append(graph_preprocess_time)

    return database
