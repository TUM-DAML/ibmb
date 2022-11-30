import numba
from numba.typed import List
import numpy as np
from scipy.sparse import find
import parallel_sort


def get_pairs(ppr_mat):
    """
    returns: re-labeled pairs
    """
    ppr_mat = ppr_mat + ppr_mat.transpose()

    ppr_mat = ppr_mat.tocoo()   # find issue: https://github.com/scipy/scipy/blob/v1.7.1/scipy/sparse/extract.py#L12-L40
    row, col, data = ppr_mat.row, ppr_mat.col, ppr_mat.data
    mask = (row > col)  # lu

    row, col, data = row[mask], col[mask], data[mask]

    # sort_arg = np.argsort(data)[::-1]
    sort_arg = parallel_sort.parallel_argsort(data)[::-1]

    # map prime_nodes to arange
    ppr_pairs = np.vstack((row[sort_arg], col[sort_arg])).T
    return ppr_pairs


@numba.njit(cache=True, locals={'p1': numba.int64, 
                                'p2': numba.int64, 
                                'p3': numba.int64, 
                                'new_list': numba.int64[::1]})
def merge_lists(lst1, lst2):
    p1, p2, p3 = numba.int64(0), numba.int64(0), numba.int64(0)
    new_list = np.zeros(len(lst1) + len(lst2), dtype=np.int64)
    
    while p2 < len(lst2) and p1 < len(lst1):
        if lst2[p2] <= lst1[p1]:
            new_list[p3] = lst2[p2]
            p2 += 1
            
            if lst2[p2 - 1] == lst1[p1]:
                p1 += 1
                
        elif lst2[p2] > lst1[p1]:
            new_list[p3] = lst1[p1]
            p1 += 1
        p3 += 1
    
    if p2 == len(lst2) and p1 == len(lst1):
        return new_list[:p3]
    elif p1 == len(lst1):
        rest = lst2[p2:]
    elif p2 == len(lst2):
        rest = lst1[p1:]
        
    p3_ = p3 + len(rest)
    new_list[p3: p3_] = rest
    
    return new_list[:p3_]


@numba.njit(cache=True)
def merge_multi_lists(lst_lists, placeholder=0):
    if len(lst_lists) <= 1:
        return lst_lists
    elif len(lst_lists) == 2:
        return [merge_lists(lst_lists[0], lst_lists[1])]
    else:
        n = len(lst_lists)
        left = merge_multi_lists(lst_lists[:n // 2])
        right = merge_multi_lists(lst_lists[n // 2:])
        return [merge_lists(left[0], right[0])]