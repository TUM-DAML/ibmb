import numpy as np
import numba
from numba.typed import Dict, List
import scipy.sparse as sp


@numba.njit(cache=True)
def gleich_hk(indptr, indices, seed, N, t, eps, psis):
    
    x = {}  # Store x, r as dictionaries
    r = {}  # initialize residual
    Q = []  # initialize queue
    for s in seed:
        r[(s, 0)] = numba.float32(1. / len(seed))
        Q.append((s, 0))
        
    while len(Q) > 0:
        (v, j) = Q.pop(0)  # v has r[(v,j)] ...
        rvj = r[(v, j)]
        # perform the hk-relax step
        if v not in x:
            x[v] = rvj
        else:
            x[v] += rvj
            
        r[(v, j)] = numba.float32(0)
        mass = numba.float32((t * rvj / (j + 1.)) / (indptr[v+1] - indptr[v]))
        
        for u in indices[indptr[v] : indptr[v+1]]:  # for neighbors of v
            _next = (u, j + 1)  # in the next block
            if j + 1 == N:
                if u not in x:
                    x[u] = numba.float32(rvj / (indptr[v+1] - indptr[v]))
                else:
                    x[u] += numba.float32(rvj / (indptr[v+1] - indptr[v]))
                continue
                
            if _next not in r:
                r[_next] = numba.float32(0)
            thresh = np.exp(t) * eps * (indptr[u+1] - indptr[u])
            thresh = thresh / (N * psis[j + 1])
            if r[_next] < thresh <= r[_next] + mass:
                Q.append(_next)  # add u to queue
            r[_next] = r[_next] + mass
    return list(x.keys()), list(x.values())


@numba.njit(cache=True, parallel=True)
def calc_hk_parallel(indptr, indices, nodes, t, eps, N, topk):
    inds = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    
    psis = np.ones(N + 1, dtype=np.float32)
    for i in range(N - 1, 0, -1):
        psis[i] = psis[i + 1] * t / (i + 1.) + 1.
        
    for i in numba.prange(len(nodes)):
        seed = List([nodes[i]])
        ind, val = gleich_hk(indptr, indices, seed, N, t, eps, psis)
        ind_np, val_np = np.array(ind), np.array(val)
        idx_topk = np.argsort(val_np)[-topk:]
        inds[i] = ind_np[idx_topk]
        vals[i] = val_np[idx_topk]
    return inds, vals


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int64))
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


def topk_hk_neighbors(adj_matrix, nodes, t=5, eps=1e-4, N=10, topk=32):
    indptr, indices = adj_matrix.indptr, adj_matrix.indices
    
    neighbors, weights = calc_hk_parallel(indptr, indices, nodes, t, eps, N, topk)
    
#     mat = construct_sparse(neighbors, weights, (len(nodes), adj_matrix.shape[0]))
#     return mat.tocsr()
    for i in range(len(nodes)):
        neighbors[i] = np.union1d(neighbors[i], nodes[i])
    return neighbors


# for notebook debugging

# from get_neighbors import topk_hk_matrix, hk_power_method
# from scipy.sparse import find

# mat = topk_hk_matrix(adj, np.arange(100), N=10, eps=1e-4, topk=100)
# pw_method = hk_power_method(graph.adj_t.cuda(), [[i] for i in range(100)], 20, t)

# for i in range(100):
#     mask_power_method = pw_method[i].argsort()[-100:]
#     mask_gleich = find(mat[i, :])
    
#     print(len(np.intersect1d(mask_power_method, mask_gleich)))