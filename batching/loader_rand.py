from .util_func import merge_lists
import numpy as np


def rand_nonfixed_loader(prime_indices,
                         neighbors,
                         merge_max_size=500):
    prim_nodes = []
    batch_nodes = np.zeros(0, dtype=np.int64)

    indices = np.random.permutation(len(prime_indices))

    for i in indices:
        node = prime_indices[i]
        merged_batch = merge_lists(batch_nodes, neighbors[i])

        if i == indices[-1]:
            batch_nodes = merged_batch
            prim_nodes.append(node)
            yield True, (np.array(prim_nodes, dtype=np.int64), batch_nodes.astype(np.int64))
        else:
            if len(merged_batch) <= merge_max_size:
                batch_nodes = merged_batch
                prim_nodes.append(node)
            else:
                yield False, (np.array(prim_nodes, dtype=np.int64), batch_nodes.astype(np.int64))
                prim_nodes = [node]
                batch_nodes = neighbors[i]


def rand_fixed_loader(prime_indices, neighbors, merge_max_size):
    gen = rand_nonfixed_loader(prime_indices, neighbors, merge_max_size)

    lst = []
    for _, batch in gen:
        lst.append(batch)
    return lst


class RandLoader:
    def __init__(self, prime_indices, neighbors, merge_max_size):
        self.fetch_idx = None
        assert len(prime_indices) == len(neighbors)
        self.max_neighbor_num = max([len(i) for i in neighbors])
        self.prime_indices = prime_indices
        self.neighbors = neighbors
        self.merge_max_size = merge_max_size

        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        # never accurate, an upperbound of num_batches
        min_num_primes_perbatch = self.merge_max_size // self.max_neighbor_num
        max_num_batches = len(self.prime_indices) // min_num_primes_perbatch + 1
        return max_num_batches

    def reset(self):
        self.fetch_idx = list(np.random.permutation(len(self.prime_indices)))

    def next(self):
        if not len(self.fetch_idx):
            self.reset()
            raise StopIteration()

        prim_nodes = []
        batch_nodes = np.zeros(0, dtype=np.int64)

        while len(self.fetch_idx):
            i = self.fetch_idx.pop(0)
            node = self.prime_indices[i]
            merged_batch = merge_lists(batch_nodes, self.neighbors[i])

            if len(merged_batch) <= self.merge_max_size:
                batch_nodes = merged_batch
                prim_nodes.append(node)
            else:
                self.fetch_idx.insert(0, i)  # put it back!
                return False, (np.array(prim_nodes, dtype=np.int64), batch_nodes.astype(np.int64))
        return True, (np.array(prim_nodes, dtype=np.int64), batch_nodes.astype(np.int64))
