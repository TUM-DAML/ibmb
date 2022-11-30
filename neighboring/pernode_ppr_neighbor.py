import numba
import numpy as np
import scipy.sparse as sp


@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon, topk=None, patience=5):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    
    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)

    return list(p.keys()), list(p.values())


# # with early stop
# @numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32, 
#                                 'inds': numba.int32[::1], 'vals': numba.float32[::1], 
#                                 'heuristic_topk': numba.int32[::1]})
# def _calc_ppr_node(inode, indptr, indices, deg, alpha, alpha_eps, topk, patience=10, isolate_patience=15):
    
#     f32_0 = numba.float32(0)
#     p = {inode: f32_0}
#     r = {}
#     r[inode] = alpha
#     q = [inode]
#     heuristic_topk = np.zeros(0, dtype=numba.int32)
#     isolate_topk = 0
#     isolate_patience_count = 0
#     patience_count = numba.int32(0)
    
#     while len(q) > 0:
#         unode = q.pop()

#         res = r[unode] if unode in r else f32_0
#         if unode in p:
#             p[unode] += res
#         else:
#             p[unode] = res
#         r[unode] = f32_0
#         for vnode in indices[indptr[unode]:indptr[unode + 1]]:
#             _val = (1 - alpha) * res / deg[unode]
#             if vnode in r:
#                 r[vnode] += _val
#             else:
#                 r[vnode] = _val

#             res_vnode = r[vnode] if vnode in r else f32_0
#             if res_vnode >= alpha_eps * deg[vnode]:
#                 if vnode not in q:
#                     q.append(vnode)
                    
        
        
# #         tops = np.argpartition(vals, kth=len(vals) - topk)[-topk:]
#         if len(p) >= topk:
            
#             inds, vals = np.array(list(p.keys()), dtype=numba.int32), \
#                     np.array(list(p.values()), dtype=numba.float32)
            
#             tops = inds[np.argsort(vals)[-topk:]]
# #         tops = np.sort(tops)
        
#             if np.array_equal(tops, heuristic_topk):
#     #             if len(tops) >= topk:
#                 patience_count += 1
#                 if patience_count >= patience:
#                     break
#     #             else:
#     #                 if len(tops) != isolate_topk:
#     #                     isolate_topk = len(tops)
#     #                     isolate_patience_count = 0
#     #                 else:
#     #                     isolate_patience_count += 1
#     #                     if isolate_patience_count >= isolate_patience:
#     #                         break
#             else:
#                 patience_count = 0
#                 heuristic_topk = tops
        
#     return list(p.keys()), list(p.values())


@numba.njit(cache=True)
def calc_ppr(indptr, indices, deg, alpha, epsilon, nodes):
    js = []
    vals = []
    alpha_eps = alpha * epsilon
    for i, node in enumerate(nodes):
        j, val = _calc_ppr_node(node, indptr, indices, deg, alpha, alpha_eps)
        js.append(j)
        vals.append(val)
    return js, vals


@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk):
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    for i in numba.prange(len(nodes)):
        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon, topk)
        j_np, val_np = np.array(j), np.array(val)
        idx_topk = np.argsort(val_np)[-topk:]
        js[i] = j_np[idx_topk]
        vals[i] = val_np[idx_topk]
    return js, vals


def ppr_topk(adj_matrix, alpha, epsilon, nodes, topk):
    """Calculate the PPR matrix approximately using Anderson."""

    out_degree = np.sum(adj_matrix > 0, axis=1).A1
    nnodes = adj_matrix.shape[0]

    neighbors, weights = calc_ppr_topk_parallel(adj_matrix.indptr, adj_matrix.indices, out_degree,
                                                numba.float32(alpha), numba.float32(epsilon), nodes, topk)

    return construct_sparse(neighbors, weights, (len(nodes), nnodes)), neighbors


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


def topk_ppr_matrix(adj_matrix, alpha, eps, idx, topk, normalization='row'):
    """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""

    topk_matrix, neighbors = ppr_topk(adj_matrix, alpha, eps, idx, topk)
    topk_matrix = topk_matrix.tocsr()

    if normalization == 'sym':
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_sqrt = np.sqrt(np.maximum(deg, 1e-12))
        deg_inv_sqrt = 1. / deg_sqrt

        row, col = topk_matrix.nonzero()
        # assert np.all(deg[idx[row]] > 0)
        # assert np.all(deg[col] > 0)
        topk_matrix.data = deg_sqrt[idx[row]] * topk_matrix.data * deg_inv_sqrt[col]
    elif normalization == 'col':
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_inv = 1. / np.maximum(deg, 1e-12)

        row, col = topk_matrix.nonzero()
        # assert np.all(deg[idx[row]] > 0)
        # assert np.all(deg[col] > 0)
        topk_matrix.data = deg[idx[row]] * topk_matrix.data * deg_inv[col]
    elif normalization == 'row':
        pass
    else:
        raise ValueError(f"Unknown PPR normalization: {normalization}")

    return topk_matrix, neighbors
