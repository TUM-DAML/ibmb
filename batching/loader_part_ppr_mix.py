import numpy as np


def partition_ppr_loader(partitions, prime_indices, neighbor_list):
    n = len(partitions)
    batches = []
    if isinstance(neighbor_list, list):
        neighbor_list = np.array(neighbor_list, dtype=object)
    for i in range(n):
        intersect = np.intersect1d(partitions[i], prime_indices)
        ind = np.in1d(prime_indices, intersect)
        lst = list(neighbor_list[ind])
        seconds = np.unique(np.concatenate(lst))
        batches.append((intersect, seconds,))
    
    return batches
