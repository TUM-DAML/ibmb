import logging

import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.data import NeighborSampler
from tqdm import tqdm

from batching.MySaintSampler import SaintRWTrainSampler, SaintRWValSampler
from . import normalize_adjmat
from .data_utils import get_pair_wise_distance
from .modified_tsp import tsp_heuristic


class MYDataset:

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 adj_t: csr_matrix,
                 train_loader: [list, NeighborSampler, SaintRWTrainSampler, SaintRWValSampler],
                 val_loader: list,
                 test_loader: list,
                 batch_order: dict,
                 cache_sub_adj: bool = True,
                 cache_origin_adj: bool = False,
                 cache: bool = True,
                 device: str = 'cuda',
                 reweight: bool = False,
                 re_normalization: str = 'sym'):
        """
        A class containing 
            training loader: partitions-based / PPR-based or sampling torch Dataloader
            val loader: the first storing the original loader, the second storing LBMB loader
            test loader: same as val loader
        
        Potentially caching the batches for faster fetching: 
            training / original val / original test cache (except neighbor / RW / LADIES sampling loader)
            val / test LBMB loader
        """
        self.edge_index_mat = adj_t

        self.x = x
        self.y = y

        num_classes = y.max() + 1

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.device = device

        self.train_cache = []
        self.val_cache_self = []
        self.val_cache_part = []
        self.val_cache_ppr = []
        self.test_cache_self = []
        self.test_cache_part = []
        self.test_cache_ppr = []

        self.cache = cache
        self.cache_sub_adj = cache_sub_adj
        self.cache_origin_adj = cache_origin_adj
        self.reweight = reweight
        self.re_normalization = re_normalization

        self.batch_kl_div = None
        self.last_train_batch_id = 0
        self.random_order = not (batch_order['ordered'] or batch_order['sampled'])

        self.set_cache()

        if cache:
            if batch_order['ordered'] or batch_order['sampled']:
                if len(self.train_cache) > 2:
                    self.batch_kl_div = get_pair_wise_distance([batch[1] for batch in self.train_cache],
                                                               num_classes,
                                                               dist_type='kl')
                    if batch_order['ordered']:
                        best_perm, _ = tsp_heuristic(self.batch_kl_div)
                        ordered_batches = [self.train_cache[i] for i in best_perm]
                        logging.info(f'best permutation: {best_perm}')
                        self.train_cache = ordered_batches
                        self.batch_kl_div = None  # no need anymore
                else:  # no need to consider order
                    pass

    def _subgraph(self, subset):
        edge_index = self.edge_index_mat[subset, :][:, subset]

        if self.reweight:
            edge_index = normalize_adjmat(edge_index, self.re_normalization)

        return edge_index

    def set_split(self, split):
        lst = ['train', 'val_self', 'val_part', 'val_ppr', 'test_self', 'test_part', 'test_ppr']
        assert split in lst

        self.split = split
        if split == lst[0]:
            self.cur_loader = self.train_loader
            self.cur_cache = self.train_cache
        elif split == lst[1]:
            self.cur_loader = self.val_loader[0]
            self.cur_cache = self.val_cache_self
        elif split == lst[2]:
            self.cur_loader = self.val_loader[1]
            self.cur_cache = self.val_cache_part
        elif split == lst[3]:
            self.cur_loader = self.val_loader[2]
            self.cur_cache = self.val_cache_ppr
        elif split == lst[4]:
            self.cur_loader = self.test_loader[0]
            self.cur_cache = self.test_cache_self
        elif split == lst[5]:
            self.cur_loader = self.test_loader[1]
            self.cur_cache = self.test_cache_part
        elif split == lst[6]:
            self.cur_loader = self.test_loader[2]
            self.cur_cache = self.test_cache_ppr

        return True if self.cur_loader is not None else False

    def clear_cur_cache(self):
        lst = ['train', 'val_self', 'val_part', 'val_ppr', 'test_self', 'test_part', 'test_ppr']
        assert self.split in lst

        if self.split == lst[0]:
            self.cur_cache = self.train_cache = []
        elif self.split == lst[1]:
            self.cur_cache = self.val_cache_self = []
        elif self.split == lst[2]:
            self.cur_cache = self.val_cache_part = []
        elif self.split == lst[3]:
            self.cur_cache = self.val_cache_ppr = []
        elif self.split == lst[4]:
            self.cur_cache = self.test_cache_self = []
        elif self.split == lst[5]:
            self.cur_cache = self.test_cache_part = []
        elif self.split == lst[6]:
            self.cur_cache = self.test_cache_ppr = []

        logging.info(f'{self.split} cache cleared!')

    def set_cache(self):

        lst = ['train', 'val_self', 'val_part', 'val_ppr', 'test_self', 'test_part', 'test_ppr'] if self.cache else \
            ['val_part', 'val_ppr', 'test_part', 'test_ppr']

        for item in lst:
            if self.set_split(item):
                logging.info(f'\n setting cache for {item} \n')
                for idx in tqdm(range(len(self.cur_loader))):
                    self.cur_cache.append(self.get_batch(idx, self.cache_sub_adj))

        if self.cache:
            self.x = None
            self.y = None
        if not self.cache_origin_adj:
            self.edge_index_mat = None

    def get_batch(self, idx, get_adj=True):
        primes, seconds = self.cur_loader[idx]
        return self._get_batch(primes, seconds, get_adj)

    def _get_batch(self, primes, seconds, get_adj=True):
        if isinstance(seconds, np.ndarray):  # already merged
            mask = np.in1d(seconds, primes).nonzero()[0]
        elif isinstance(seconds, list):  # typically for shadow loader, where subgraphs do not overlap
            mask = np.concatenate([sec == prm for prm, sec in zip(primes, seconds)], axis=0)
            seconds = np.concatenate(seconds, axis=0)
        else:
            raise TypeError

        if get_adj:
            adj = self._subgraph(seconds)
            adj_data = adj.indptr.astype(np.int64), adj.indices.astype(np.int64), adj.data
        else:
            adj_data = None

        x = self.x[seconds]
        y = self.y[seconds][mask]

        return x, y, adj_data, mask

    def __len__(self):
        return len(self.cur_loader) if self.cur_loader is not None else 0

    def __getitem__(self, idx):
        if self.cache or (self.split in ['val_part', 'val_ppr', 'test_part', 'test_ppr']):
            return self.cur_cache[idx]
        else:
            return self.get_batch(idx)
