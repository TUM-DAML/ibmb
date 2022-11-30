import numpy as np
from tqdm import tqdm


# @numba.njit(cache=True, parallel=True)
def partition_ppr_loader(partitions, prime_indices, neighbor_list):
    n = len(partitions)
    batches = []
    # for i in numba.prange(len(nodes)):
    for i in range(n):
        intersect = np.intersect1d(partitions[i], prime_indices)
        ind = np.in1d(prime_indices, intersect)
        lst = list(neighbor_list[ind])
        seconds = np.unique(np.concatenate(lst))
        batches.append((intersect, seconds,))
    
    return batches
