import queue
import threading

import numpy as np
import torch
from torch_sparse import SparseTensor

from data.customed_dataset import MYDataset


def get_prefetch_generator(mode: str,
                           neighbor_sampling: str,
                           dataset: MYDataset,
                           prime_nodes: np.ndarray = None,
                           batch_size: int = 1,
                           max_prefetch: int = 1):
    if dataset.split in ['train', 'val_self', 'test_self']:
        loader = dataset.cur_loader
        if 'ladies' in [mode, neighbor_sampling]:
            return LADIESGenerator(dataset.x,
                                   dataset.y,
                                   loader,
                                   max_prefetch=max_prefetch)

        elif mode == 'n_sampling':
            return NeighborSampleGenerator(loader,
                                           dataset.x,
                                           dataset.y,
                                           max_prefetch=max_prefetch)
        elif mode == 'rw_sampling':
            return RWGenerator(loader,
                               prime_nodes,
                               dataset.x,
                               dataset.y,
                               max_prefetch=max_prefetch)
        elif mode in ['rand', 'ppr_shadow']:
            return RandBatchingGenerator(dataset, max_prefetch=max_prefetch)

    # else: LBMB inference or training
    if dataset.split != 'train':
        lst = np.arange(len(dataset))
    else:
        if dataset.random_order:
            lst = np.random.permutation(len(dataset))
        else:
            if dataset.batch_kl_div is None:  # optim
                lst = np.arange(len(dataset))
            else:
                lst = None

    return BackgroundGenerator(lst,
                               dataset,
                               batch_size=batch_size,
                               max_prefetch=max_prefetch)


class BaseGenerator(threading.Thread):
    def __init__(self, max_prefetch=1, device='cuda'):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.device = device
        self.stop_signal = False
        self.start()

    def run(self):
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class BackgroundGenerator(BaseGenerator):
    def __init__(self, fetch_idx, dataset, batch_size=1, max_prefetch=1, device='cuda'):

        self.dataset = dataset
        self.batch_size = batch_size

        if fetch_idx is not None:
            self.fetch_idx = fetch_idx

        else:
            probs = dataset.batch_kl_div.copy()

            last = dataset.last_train_batch_id
            num_batches = probs.shape[0]

            self.fetch_idx = []

            next_id = 0
            while np.any(probs):
                # sampling 
                next_id = np.random.choice(num_batches, size=None, replace=False, p=probs[last] / probs[last].sum())
                # greedy
                # next_id = np.argmax(probs[last])
                last = next_id
                self.fetch_idx.append(next_id)
                probs[:, next_id] = 0.

            dataset.last_train_batch_id = next_id

        super().__init__(max_prefetch, device)

    def run(self):

        indptr = [np.zeros(1, dtype=np.int64)]
        indices = []
        data = []
        last_mask = []
        batch_x = []
        batch_y = []
        length = 0
        sparse_entries = 0
        secondary_nodes = []

        for n, i in enumerate(self.fetch_idx):
            if self.stop_signal:
                break
            x, y, edge_index, mask = self.dataset[i]
            batch_x.append(x)
            batch_y.append(y)
            last_mask.append(mask + length)

            if edge_index is not None:  # doesn't contain inter-cluster edges
                indptr.append(edge_index[0][1:] + sparse_entries)
                indices.append(edge_index[1] + length)
                data.append(edge_index[2])
                sparse_entries += data[-1].shape[0]
            else:  # contains inter-cluster edges
                _, second = self.dataset.cur_loader[i]

                secondary_nodes.append(second)

            length += batch_x[-1].shape[0]

            if (n + 1) % self.batch_size == 0 or n == len(self.dataset) - 1:
                last_mask = np.concatenate(last_mask, axis=0)

                if len(data):
                    indptr = np.concatenate(indptr, axis=0)
                    indices = np.concatenate(indices, axis=0)
                    data = np.concatenate(data, axis=0)
                else:
                    if len(secondary_nodes):
                        secondary_nodes = np.concatenate(secondary_nodes, axis=0)
                        adj = self.dataset._subgraph(secondary_nodes)
                        indptr, indices, data = adj.indptr.astype(np.int64), adj.indices.astype(np.int64), adj.data
                    else:
                        raise ValueError

                batch_x = np.concatenate(batch_x, axis=0)
                batch_y = np.concatenate(batch_y, axis=0)

                if self.stop_signal:
                    break

                self.queue.put(((torch.from_numpy(batch_x).to(self.device, non_blocking=True),
                                 torch.from_numpy(batch_y).to(self.device, non_blocking=True),
                                 SparseTensor(rowptr=torch.from_numpy(indptr).to(self.device, non_blocking=True),
                                              col=torch.from_numpy(indices).to(self.device, non_blocking=True),
                                              value=torch.from_numpy(data).to(self.device, non_blocking=True),
                                              sparse_sizes=(length, length)),
                                 torch.from_numpy(last_mask).to(self.device, non_blocking=True), None,),
                                i == self.fetch_idx[-1]))

                length = 0
                sparse_entries = 0
                indptr = [np.zeros(1, dtype=np.int64)]
                indices = []
                data = []
                last_mask = []
                batch_x = []
                batch_y = []
                secondary_nodes = []

        self.queue.put(None)


class NeighborSampleGenerator(BaseGenerator):
    def __init__(self, dataloader, x, y, max_prefetch=1, device='cuda'):

        self.dataloader = dataloader
        self.x = x
        self.y = y
        super().__init__(max_prefetch, device)

    def run(self):

        for i, (batch_size, n_id, adj) in enumerate(self.dataloader):
            if self.stop_signal:
                break
            n_id = n_id.cpu().numpy()
            y = torch.from_numpy(self.y[n_id[:batch_size]]).to(self.device, non_blocking=True)
            x = torch.from_numpy(self.x[n_id]).to(self.device, non_blocking=True)
            adjs = [a.adj_t.to(self.device, non_blocking=True) for a in adj]
            if self.stop_signal:
                break
            self.queue.put(((x, y, adjs, None, None), i == len(self.dataloader) - 1))

        self.queue.put(None)


class LADIESGenerator(BaseGenerator):
    def __init__(self, x, y, dataloader, max_prefetch=1, device='cuda'):

        self.dataloader = dataloader
        self.x = x
        self.y = y

        super().__init__(max_prefetch, device)

    def run(self):
        for flag, (edges, node_indices) in self.dataloader:
            if self.stop_signal:
                self.dataloader.reset()
                break

            adjs = []
            for indptr, indices, data, size in edges:
                adjs.append(SparseTensor(rowptr=torch.from_numpy(indptr).to(self.device, non_blocking=True),
                                         col=torch.from_numpy(indices).to(self.device, non_blocking=True),
                                         value=torch.from_numpy(data).to(self.device, non_blocking=True),
                                         sparse_sizes=size))

            y = torch.from_numpy(self.y[node_indices[-1]]).to(self.device, non_blocking=True)

            x = torch.from_numpy(self.x[node_indices[0]]).to(self.device, non_blocking=True)

            if self.stop_signal:
                self.dataloader.reset()
                break
            self.queue.put(((x, y, adjs, None, None), flag))

        self.queue.put(None)


class RWGenerator(BaseGenerator):
    def __init__(self, dataloader, prime_nodes, x, y, max_prefetch=1, device='cuda'):

        self.dataloader = dataloader
        self.prime_nodes = prime_nodes
        self.x = x
        self.y = y

        super().__init__(max_prefetch, device)

    def run(self):
        len_loader = self.dataloader.loader_len()

        for i, ((prime_nodes, node_idx), adj, node_norm) in enumerate(self.dataloader):
            #             if self.stop_signal:
            #                 break
            node_idx = node_idx.numpy()
            if prime_nodes is None:
                mask = np.in1d(node_idx, self.prime_nodes).nonzero()[0]  # sample with overlapping prime nodes
            else:
                prime_nodes = prime_nodes.numpy()
                mask = np.in1d(node_idx, prime_nodes).nonzero()[0]

            y = torch.from_numpy(self.y[node_idx][mask]).to(self.device, non_blocking=True)
            x = torch.from_numpy(self.x[node_idx]).to(self.device, non_blocking=True)
            adj = adj.to(self.device, non_blocking=True)
            mask = torch.from_numpy(mask).to(self.device, non_blocking=True)

            node_norm = node_norm.to(self.device, non_blocking=True)
            if prime_nodes is None:
                node_norm = node_norm[mask]
            node_norm /= node_norm.sum()

            #             if self.stop_signal:
            #                 break
            self.queue.put(((x, y, adj, mask, node_norm), i == len_loader - 1))

        self.queue.put(None)


class RandBatchingGenerator(BaseGenerator):
    def __init__(self, dataset, max_prefetch=1, device='cuda'):
        self.dataset = dataset
        super().__init__(max_prefetch, device)

    def run(self):
        for last_flag, (p, n) in self.dataset.cur_loader:
            if self.stop_signal:
                self.dataset.cur_loader.reset()
                break

            x, y, (indptr, indices, data,), mask = self.dataset._get_batch(p, n)
            self.queue.put(((torch.from_numpy(x).to(self.device, non_blocking=True),
                             torch.from_numpy(y).to(self.device, non_blocking=True),
                             SparseTensor(rowptr=torch.from_numpy(indptr).to(self.device, non_blocking=True),
                                          col=torch.from_numpy(indices).to(self.device, non_blocking=True),
                                          value=torch.from_numpy(data).to(self.device, non_blocking=True),
                                          sparse_sizes=(len(x), len(x))),
                             torch.from_numpy(mask).to(self.device, non_blocking=True), None,),
                            last_flag))

            if self.stop_signal:
                self.dataset.cur_loader.reset()
                break
        self.queue.put(None)
