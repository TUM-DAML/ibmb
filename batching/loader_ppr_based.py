from .util_func import merge_lists, merge_multi_lists, get_pairs
import numba
import time
from numba.typed import List
import numpy as np
import queue as Q
import logging


# =============================================================================================
# The clusters should be

# sorted
# int64


def post_process(loader, merge_max_size):
    # merge smallest clusters first
    que = Q.PriorityQueue()
    for p, n in loader:
        que.put((len(n), (list(p), list(n))))
    
    while que.qsize() > 1:
        len1, (p1, n1) = que.get()
        len2, (p2, n2) = que.get()
        n = merge_lists(np.array(n1), np.array(n2))

        if len(n) > merge_max_size:
            que.put((len1, (p1, n1)))
            que.put((len2, (p2, n2)))
            break

        else:
            que.put((len(n), (p1 + p2, list(n))))
    
    new_batch = []
    
    while not que.empty():
        _, (p, n) = que.get()
        new_batch.append( (np.array(p), np.array(n)) )
    
    return new_batch


@numba.njit(cache=True, locals={'node_id_list': numba.int64[::1], 
                                'placeholder': numba.int64[::1], 
                                'id1': numba.int64, 
                                'id2': numba.int64})
def get_loader(ppr_pairs, prime_indices, id_second_list, merge_max_size=100000):
    # every batch has a unique id
    # every node belongs to a batch
    
    thresh = numba.int64(merge_max_size * 1.0005)
    
    num_nodes = len(prime_indices)
    node_id_list = np.arange(num_nodes, dtype=np.int64)
    
    id_prime_list = List(np.arange(num_nodes, dtype=np.int64).reshape(-1, 1))
    size_flag = [{a} for a in np.arange(num_nodes, dtype=np.int64)]
    
    placeholder = np.zeros(0, dtype=np.int64)
    
#     accs = []
#     decs = []
#     merge_cnt = 0
        
    for (n1, n2) in ppr_pairs:
        id1, id2 = node_id_list[n1], node_id_list[n2]
        id1, id2 = (id1, id2) if id1 < id2 else (id2, id1)
        
        if id1 != id2 and not (id2 in size_flag[id1]) and not (id1 in size_flag[id2]):
            
            batch_second1 = id_second_list[id1]
            batch_second2 = id_second_list[id2]
            
            if len(batch_second1) + len(batch_second2) <= thresh:
                new_batch_second = merge_lists(batch_second1, batch_second2)
                
#                 merge_cnt += 1
#                 ratio = (len(batch_second1) + len(batch_second2)) / merge_max_size

                if len(new_batch_second) <= merge_max_size:

#                     accs.append(ratio)

                    batch_prime1 = id_prime_list[id1]
                    batch_prime2 = id_prime_list[id2]

                    new_batch_prime = np.concatenate((batch_prime1, batch_prime2))

                    id_prime_list[id1] = new_batch_prime
                    id_second_list[id1] = new_batch_second
                    id_second_list[id2] = placeholder

                    id_prime_list[id2] = placeholder

                    node_id_list[batch_prime2] = id1
                    size_flag[id1].update(size_flag[id2])
                    size_flag[id2].clear()

                else:
#                     decs.append(ratio)

                    size_flag[id1].add(id2)
                    size_flag[id2].add(id1)
                    
            else:
                size_flag[id1].add(id2)
                size_flag[id2].add(id1)
    
    prime_second_lst = List()
    ids = np.unique(node_id_list)
    
    for _id in ids:
        prime_second_lst.append((prime_indices[id_prime_list[_id]], 
                                 id_second_list[_id]))
    
#     return list(prime_second_lst), accs, decs, merge_cnt
    return list(prime_second_lst)


# layz merge
# @numba.njit(cache=True, locals={'node_id_list': numba.int64[::1], 
#                                 'placeholder': numba.int64[::1], 
#                                 'lens': numba.int64[::1], 
#                                 'id1': numba.int64, 
#                                 'id2': numba.int64})
# def get_loader(ppr_pairs, prime_indices, id_second_list, merge_max_size=100000):
    
#     num_nodes = len(prime_indices)
#     node_id_list = np.arange(num_nodes, dtype=np.int64)
    
#     id_prime_list = List(np.arange(num_nodes, dtype=np.int64).reshape(-1, 1))
#     size_flag = [{a} for a in np.arange(num_nodes, dtype=np.int64)]
    
#     placeholder = np.zeros(0, dtype=np.int64)
    
#     id_lens = np.zeros(num_nodes, dtype=np.int64)
    
#     _id_second_list = []
    
#     for i, batch in enumerate(id_second_list):
#         id_lens[i] = len(batch)
#         _id_second_list.append([batch])
#         id_second_list[i] = placeholder
#     id_second_list = None
    
#     for (n1, n2) in ppr_pairs:
#         id1, id2 = node_id_list[n1], node_id_list[n2]
#         id1, id2 = (id1, id2) if id1 < id2 else (id2, id1)
        
#         if id1 != id2 and not (id2 in size_flag[id1]) and not (id1 in size_flag[id2]):
            
#             if id_lens[id1] + id_lens[id2] <= merge_max_size:
#                 # update id
#                 node_id_list[id_prime_list[id2]] = id1
#                 size_flag[id1].update(size_flag[id2])
#                 size_flag[id2].clear()
                
#                 # merge prime nodes
#                 id_prime_list[id1] = np.concatenate((id_prime_list[id1], id_prime_list[id2]))
#                 id_prime_list[id2] = placeholder
                
#                 # temporarily merge second nodes
#                 _id_second_list[id1] += _id_second_list[id2]
#                 _id_second_list[id2] = [placeholder]
                
#                 # update aux information
#                 id_lens[id1] += id_lens[id2]
#                 id_lens[id2] = 0
                
#             else:
#                 _id_second_list[id1] = merge_multi_lists(_id_second_list[id1], id_lens[id1])
#                 _id_second_list[id2] = merge_multi_lists(_id_second_list[id2], id_lens[id2])
#                 id_lens[id1] = len(_id_second_list[id1][0])
#                 id_lens[id2] = len(_id_second_list[id2][0])
                
#                 merged_list = merge_lists(_id_second_list[id1][0], _id_second_list[id2][0])
#                 if len(merged_list) <= merge_max_size:
#                     # update id
#                     node_id_list[id_prime_list[id2]] = id1
#                     size_flag[id1].update(size_flag[id2])
#                     size_flag[id2].clear()

#                     # merge prime nodes
#                     id_prime_list[id1] = np.concatenate((id_prime_list[id1], id_prime_list[id2]))
#                     id_prime_list[id2] = placeholder

#                     # temporarily merge second nodes
#                     _id_second_list[id1] = [merged_list]
#                     _id_second_list[id2] = [placeholder]

#                     # update aux information
#                     id_lens[id1] = len(merged_list)
#                     id_lens[id2] = 0
#                 else:
#                     size_flag[id1].add(id2)
#                     size_flag[id2].add(id1)
    
#     prime_second_lst = List()
#     ids = np.unique(node_id_list)
    
#     for _id in ids:
#         prime_second_lst.append((prime_indices[id_prime_list[_id]], 
#                                  merge_multi_lists(_id_second_list[_id])[0]))
    
#     return list(prime_second_lst)


def ppr_fixed_loader(ppr_mat, prime_indices, id_second_list, merge_max_size=100000):
    s1 = time.time()
    ppr_pairs = get_pairs(ppr_mat)
    s2 = time.time()
    logging.info('Loading pairs from mat takes {:.3f}'.format(s2 - s1))
    
    prime_second_lst = get_loader(ppr_pairs, 
                                  prime_indices, 
                                  List(id_second_list), 
                                  merge_max_size=merge_max_size)
    s3 = time.time()
    logging.info('Batching from pairs takes {:.3f}'.format(s3 - s2))
    
    # calc how many nodes are in large batches already
    n_primes = 0
    rest_primes = 0
    size_thresh = 0.8
    for p, n in prime_second_lst:
        if len(n) >= int(size_thresh * merge_max_size):
            n_primes += len(p)
        else:
            rest_primes += len(p)
    logging.info(f'{n_primes} / {len(prime_indices)} primary nodes are in batches >= {size_thresh} * size_limit')
    logging.info(f'{rest_primes} / {len(prime_indices)} primary nodes are dispersed')
    
    prime_second_lst = post_process(prime_second_lst, merge_max_size)
    s4 = time.time()
    logging.info('Merging rest takes {:.3f}'.format(s4 - s3))
    
    return prime_second_lst
