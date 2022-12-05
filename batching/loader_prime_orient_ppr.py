from .util_func import get_pairs
import numpy as np
import numba
from numba.typed import List
import queue as Q
from heapq import heappush, heappop, heapify


# @numba.njit(cache=True)
def post_process(loader, merge_max_size):
    h = [(len(p), p,) for p in loader]
    heapify(h)
    
    while len(h) > 1:
        len1, p1 = heappop(h)
        len2, p2 = heappop(h)
        if len1 + len2 <= merge_max_size:
            heappush(h, (len1 + len2, p1 + p2))
        else:
            heappush(h, (len1, p1,))
            heappush(h, (len2, p2,))
            break
    
    new_batch = []
    
    while len(h):
        _, p = heappop(h)
        new_batch.append(p)
    
    return new_batch


@numba.njit(cache=True, locals={'node_id_list': numba.int32[::1],
                                'placeholder': numba.int32[::1], })
def prime_orient_merge(ppr_pairs: np.ndarray, primes_per_batch: int, num_nodes: int):
    """
    Args:
        ppr_pairs: (# pairs, 2)
        primes_per_batch: limit of primes per batch
        
    Returns: 
        List[List]: (re-labeled) primes per batch
    """
    
    # cannot use list for id_primes_list, updating node_id_list[id_primes_list[id2]] require id_primes_list to be array
    id_primes_list = List(np.arange(num_nodes, dtype=np.int32).reshape(-1, 1))
    node_id_list = np.arange(num_nodes, dtype=np.int32)
    placeholder = np.zeros(0, dtype=np.int32)
    # size_flag = [{a} for a in np.arange(num_nodes, dtype=np.int32)]
    
    for i, j in ppr_pairs:
        id1, id2 = node_id_list[i], node_id_list[j]
        if id1 > id2:
            id1, id2 = id2, id1
        
        # if not (id1 in size_flag[id2] or id2 in size_flag[id1])
        if id1 != id2 and len(id_primes_list[id1]) + len(id_primes_list[id2]) <= primes_per_batch:
            id_primes_list[id1] = np.concatenate((id_primes_list[id1], id_primes_list[id2]))
            node_id_list[id_primes_list[id2]] = id1
            # node_id_list[j] = id1
            id_primes_list[id2] = placeholder
            
    prime_lst = List()
    ids = np.unique(node_id_list)
    
    for _id in ids:
        prime_lst.append(list(id_primes_list[_id]))
    
    return list(prime_lst)


def ppr_fixed_loader(ppr_mat, prime_indices, neighbors, primes_per_batch):
    """
    Args:
        ppr_mat: scipy.csr_matrix
        prime_indices: np.ndarray
        neighbors: List[np.ndarray] containing topk neighbors for each primary node
        num_batches: int, expected number of batches
    
    Returns:
        List[Tuple[np.ndarray]]: each tuple contains an array of primary nodes and one of aux nodes
    """
    ppr_pairs = get_pairs(ppr_mat)
    
    prime_list = prime_orient_merge(ppr_pairs, primes_per_batch, len(prime_indices))
    prime_list = post_process(prime_list, primes_per_batch)
    batches = []
    
    if isinstance(neighbors, list):
        neighbors = np.array(neighbors, dtype=object)
        
    union = lambda inputs: np.unique(np.concatenate(inputs))
    for p in prime_list:
        batches.append((prime_indices[p], union(neighbors[p]).astype(np.int64)))
    
    return batches
