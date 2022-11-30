import torch
import numpy as np
from torch_sparse import spspmm, transpose, SparseTensor
import numba
from torch_scatter import segment_coo

from data.const import ppr_iter_prams
from .pernode_ppr_neighbor import construct_sparse


def get_topk_neighbors_mask(
    num_nodes, index, ppr_scores, topk
):
    """
    Get mask that filters the PPR scores so that each node has at most `topk` neighbors.
    Assumes that `index` is sorted.
    """
    device = index.device

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_nodes)
    max_num_neighbors = num_neighbors.max()

    # Create a tensor of size [num_nodes, max_num_neighbors] to sort the PPR scores of the neighbors.
    # Use zeros so we can easily remove unused PPR scores later.
    ppr_sort = torch.zeros(
        [num_nodes * max_num_neighbors], device=device
    )

    # Create an index map to map PPR scores from ppr_scores to ppr_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    ppr_sort.index_copy_(0, index_sort_map, ppr_scores)
    ppr_sort = ppr_sort.view(num_nodes, max_num_neighbors)

    # Sort neighboring nodes based on PPR score
    ppr_sort, index_sort = torch.sort(ppr_sort, dim=1, descending=True)
    # Select the topk neighbors that are closest
    ppr_sort = ppr_sort[:, :topk]
    index_sort = index_sort[:, :topk]

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, topk
    )
    # Remove "unused pairs" with PPR scores 0
    mask_nonzero = ppr_sort > 0
    index_sort = torch.masked_select(index_sort, mask_nonzero)

    # At this point index_sort contains the index into index of the
    # closest topk neighbors per node
    # Create a mask to remove all pairs not in index_sort
    mask_topk = torch.zeros(len(index), device=device, dtype=bool)
    mask_topk.index_fill_(0, index_sort, True)

    return mask_topk


@numba.njit(cache=True, locals={'mask': numba.boolean[::1]})
def sort_coo(i, edge_index, val, topk, dim=0):
    mask = edge_index[1 - dim] == i
    sort_idx = np.argsort(val[mask])[-topk:]
    return np.sort(edge_index[dim][mask][sort_idx])


@numba.njit(cache=True, parallel=True)
def postprocess(edge_index, val, _size, topk):
    neighbors = [np.zeros(0, dtype=numba.int32)] * _size[1]
    
    for i in numba.prange(_size[1]):
        neighbors[i] = sort_coo(i, edge_index, val, topk)
    
    return neighbors


# slicing row-wise then transpose is faster than slicing col-wise
def chunk_csr_col(mat, chunksize, device='cuda', axis='col', return_adj=False):
    num_chunks = mat.size(0) // chunksize + int(mat.size(0) % chunksize > 0)
    
    for i in range(num_chunks):
        part_mat = mat[i * chunksize : min((i + 1) * chunksize, mat.size(0)), :]
        if return_adj:
            part_mat = part_mat.to(device)
            if axis == 'col':
                part_mat = part_mat.t()
            yield part_mat
            
        else:
            idx = torch.stack((part_mat.storage._row, part_mat.storage._col), 0).to(device)
            val = part_mat.storage._value.to(device)
        
            if axis == 'col':
                idx, val = transpose(idx, val, *tuple(part_mat.sizes()))
        
            yield (idx, val, part_mat.size(0))


def spspadd(index1, value1, index2, value2, sizes):
    idx = torch.cat((index1, index2), dim=1)
    val = torch.cat((value1, value2), dim=0)
    mat = torch.sparse_coo_tensor(idx, val, sizes).coalesce()
    return mat.indices(), mat.values()


def down_sample_adj(adj, topk):
    avg_num_neighbor = adj.nnz() // adj.sizes()[0]
    if topk < avg_num_neighbor:
        down_sampl_ratio = topk / avg_num_neighbor
        adj_val_np = adj.storage._value.cpu().detach().numpy()
        idx = - int(len(adj_val_np) * down_sampl_ratio)
        first_thresh = np.partition(adj_val_np, kth = idx)[idx]
        mask = (adj.storage._value >= first_thresh) | (adj.storage._row == adj.storage._col)
        adj = SparseTensor(row = adj.storage._row[mask], 
                           col = adj.storage._col[mask], 
                           value = adj.storage._value[mask], sparse_sizes=adj.sizes())
    return adj
    

# # ratio
# def ppr_power_iter(adj, dataset_name, topk, device='cuda', alpha=None, thresh=None):
    
#     chunksize, default_alpha, iters, top_percent, thresh = ppr_iter_prams[dataset_name].values()
    
#     if alpha is None:
#         alpha = default_alpha
    
#     neighbor_list = []
#     weights_list = []

#     index1 = torch.stack((adj.storage._row, adj.storage._col), 0).to(device)
#     value1 = adj.storage._value.to(device) * (1 - alpha)
#     parts = chunk_csr_col(adj, chunksize=chunksize, device='cpu')
    
#     pbar = tqdm(range(len(parts)))
#     for i in pbar:
#         index2, value2, size1 = parts[i]

#         ## push flow
#         with torch.no_grad():
#             index2, value2 = index2.to(device), value2.to(device)
#             # after first iter
#             mask = torch.where(index2[0] == (index2[1] + i * chunksize))[0]
#             p_idx = index2[:, mask]
#             p_val = torch.full(mask.size(), alpha, device=value2.device, dtype=value2.dtype)
#             if top_percent[0] < 1:
#                 mask2 = torch.topk(value2, k=int(len(value2) * top_percent[0]), sorted=False).indices
#                 mask = torch.cat((mask, mask2), dim=0).unique(sorted=True)
#                 index2, value2 = index2[:, mask], value2[mask]
#             value2 *= alpha

#             for it in range(1, iters + 1):
#                 p_idx, p_val = spspadd(p_idx, p_val, index2, value2, (adj.size(0), size1))

#                 if it < iters:
#                     index2, value2 = spspmm(index1, value1, index2, value2, adj.size(0), adj.size(1), size1)
#     #                 value2 *= 1 - alpha

#                     # identity matrix indices
#                     mask1 = torch.where(index2[0] == (index2[1] + i * chunksize))[0]
#                     mask2 = torch.topk(value2, k=int(len(value2) * top_percent[it]), sorted=False).indices
#                     mask = torch.cat((mask1, mask2), dim=0).unique(sorted=True)  # force seed node in the picked ones
#                     index2, value2 = index2[:, mask], value2[mask]

#         index_ascending, sorting_indices = torch.sort(p_idx[1])  # col should be sorted
#         ppr_scores_ascending = p_val[sorting_indices]
#         mask = get_topk_neighbors_mask(size1, index_ascending, ppr_scores_ascending, topk)

#         (row, col), val = transpose(p_idx, p_val, adj.size(0), size1)
#         split_idx = ((row[1:] > row[:-1]).nonzero().squeeze() + 1).cpu().numpy()
#         mask_splits = np.array_split(mask.cpu().numpy(), split_idx)
#         col_splits = np.array_split(col.cpu().numpy(), split_idx)
#         val_splits = np.array_split(val.cpu().numpy(), split_idx)
#     #     row_splits = np.array_split(row.cpu().numpy(), split_idx)

#         neighbor_list += [c[m] for c, m in zip(col_splits, mask_splits)]
#         weights_list += [v[m] for v, m in zip(val_splits, mask_splits)]

#         index2, value2 = None, None
#         p_idx, p_val = None, None
#         parts[i] = None

#         torch.cuda.empty_cache()
    
    

#         pbar.set_postfix(max_memory = torch.cuda.max_memory_allocated(), 
#                          memory = torch.cuda.memory_allocated(), 
#                          max_memory_reserve = torch.cuda.max_memory_reserved(), 
#                          memory_reserve = torch.cuda.memory_reserved())

#     index1 = index1.to('cpu')
#     value1 = value1.to('cpu')
    
#     torch.cuda.reset_peak_memory_stats()
    
#     pprmat = construct_sparse(neighbor_list, weights_list, adj.sizes()).tocsr()
    
#     return np.array(neighbor_list, dtype=object), pprmat


# threshold
def ppr_power_iter(adj, dataset_name, topk, device='cuda', alpha=None, thresh=None):
    
    chunksize, default_alpha, iters, top_percent, default_thresh = ppr_iter_prams[dataset_name].values()
    
    if alpha is None:
        alpha = default_alpha
    
    if thresh is None:
        thresh = default_thresh
    
    neighbor_list = []
    weights_list = []

    index1 = torch.stack((adj.storage._row, adj.storage._col), 0).to(device)
    value1 = adj.storage._value.to(device) * (1 - alpha)
    adj = down_sample_adj(adj, topk)
    parts = chunk_csr_col(adj, chunksize=chunksize, device=device)
    
    for i, (index2, value2, size1) in enumerate(parts):
        
        ## push flow
        with torch.no_grad():
            # index2, value2 = index2.to(device), value2.to(device)
            # after first iter
            mask = torch.where(index2[0] == (index2[1] + i * chunksize))[0]
            p_idx = index2[:, mask]
            p_val = torch.full(mask.size(), alpha, device=value2.device, dtype=value2.dtype)
            value2 *= alpha

            for it in range(1, iters + 1):
                p_idx, p_val = spspadd(p_idx, p_val, index2, value2, (adj.size(0), size1))

                if it < iters:
                    index2, value2 = spspmm(index1, value1, index2, value2, adj.size(0), adj.size(1), size1)
                    mask = (index2[0] == (index2[1] + i * chunksize)) | (value2 >= thresh)
                    index2, value2 = index2[:, mask], value2[mask]

        index2, value2 = None, None
        index_ascending, sorting_indices = torch.sort(p_idx[1])  # col should be sorted
        ppr_scores_ascending = p_val[sorting_indices]
        mask = get_topk_neighbors_mask(size1, index_ascending, ppr_scores_ascending, topk)

        (row, col), val = transpose(p_idx, p_val, adj.size(0), size1)
        split_idx = ((row[1:] > row[:-1]).nonzero().squeeze() + 1).cpu().numpy()
        mask_splits = np.array_split(mask.cpu().numpy(), split_idx)
        col_splits = np.array_split(col.cpu().numpy(), split_idx)
        val_splits = np.array_split(val.cpu().numpy(), split_idx)

        neighbor_list += [c[m] for c, m in zip(col_splits, mask_splits)]
        weights_list += [v[m] for v, m in zip(val_splits, mask_splits)]

        p_idx, p_val = None, None

        torch.cuda.empty_cache()

        print(f'max_memory = {torch.cuda.max_memory_allocated()}, '
                         f'memory = {torch.cuda.memory_allocated()}, '
                         f'max_memory_reserve = {torch.cuda.max_memory_reserved()}, '
                         f'memory_reserve = {torch.cuda.memory_reserved()}')

    index1 = index1.to('cpu')
    value1 = value1.to('cpu')
    
    torch.cuda.reset_peak_memory_stats()
    
    pprmat = construct_sparse(neighbor_list, weights_list, adj.sizes()).tocsr()
    
    return np.array(neighbor_list, dtype=object), pprmat
