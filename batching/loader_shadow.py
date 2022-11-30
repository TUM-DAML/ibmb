import numpy as np
from math import ceil

                
class RandShadowLoader():
    def __init__(self, prime_indices, neighbors, primes_per_batch):
        assert len(prime_indices) == len(neighbors)
        self.prime_indices = prime_indices
        self.neighbors = neighbors
        self.primes_per_batch = primes_per_batch
        
        self.reset()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
    
    def __len__(self):
        return int(ceil(len(self.prime_indices) / self.primes_per_batch))
    
    def reset(self):
        self.fetch_idx = list(np.random.permutation(len(self.prime_indices)))
        
    def next(self):
        if not len(self.fetch_idx):
            self.reset()
            raise StopIteration()
            
        prim_nodes = []
        batch_nodes = []
        
        while len(self.fetch_idx):
            i = self.fetch_idx.pop(0)
            
            if len(prim_nodes) < self.primes_per_batch:
                batch_nodes.append(self.neighbors[i])
                prim_nodes.append(self.prime_indices[i])
            else:
                return (False, (np.array(prim_nodes, dtype=np.int64), batch_nodes))
        return (True, (np.array(prim_nodes, dtype=np.int64), batch_nodes))
