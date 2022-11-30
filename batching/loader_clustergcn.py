import numpy as np
from tqdm import tqdm


def ClusterGCN_loader(partitions, prime_indices):
    loader = []
    for n, second_nodes in enumerate(tqdm(partitions)):
        primes_in_part = np.intersect1d(second_nodes, prime_indices)
        if len(primes_in_part):
            loader.append((primes_in_part, second_nodes))
    
    return loader
