from typing import List, Union

import torch
from torch_sparse import SparseTensor, matmul
from tqdm import tqdm


def get_chunk_idx(_size, num_chunks):
    chunk_size = _size // num_chunks + (_size % num_chunks > 0)
    idx = list(range(0, _size, chunk_size))
    idx.append(_size)

    return list(zip(idx[:-1], idx[1:]))


def chunked_matmul_beta(w: torch.tensor, x: torch.tensor, num_chunks):
    """
    x @ w -> [x1; x2; x3; ...] @ w
    """
    device = x.device
    idx = get_chunk_idx(x.shape[0], num_chunks)

    inplace = w.shape[0] == w.shape[1]
    if not inplace:
        new_x = torch.empty(x.shape[0], w.shape[1], dtype=x.dtype)

    with torch.no_grad():
        pbar = tqdm(idx, desc="matmul")
        for s, e in pbar:
            if inplace:
                x[s: e, :] = (x[s: e, :].to(w.device) @ w).to(device)
            else:
                new_x[s: e, :] = (x[s: e, :].to(w.device) @ w).to(device)

    if not inplace:
        return new_x
    else:
        return x


def chunk_adj_row(adj: SparseTensor, num_chunks: int) -> List[SparseTensor]:
    idx = get_chunk_idx(adj.sizes()[0], num_chunks)
    new_adj = []

    for s, e in tqdm(idx, desc="chunk adj"):
        new_adj.append(adj[s: e, :])

    return new_adj


def chunked_sp_matmul_beta(adj: List[SparseTensor],
                           x: torch.tensor,
                           num_chunks: int,
                           reduce: str = 'add',
                           device: torch.device = torch.device('cuda')):
    """
    adj @ x -> [adj1; adj2; adj3; ...] @ [x1, x2, x3, ...]
    """
    original_device = x.device
    idx = get_chunk_idx(x.shape[1], num_chunks)

    lens_adj = []
    for a in adj:
        if a is not None:
            lens_adj.append(a.sizes()[0])
        else:
            lens_adj.append(0)

    inplace = sum(lens_adj) == x.shape[0]
    if not inplace:
        new_x = torch.empty(sum(lens_adj), x.shape[1], dtype=x.dtype)

    with torch.no_grad():
        pbar = tqdm(idx, desc="spmm")
        for s, e in pbar:
            col_x = x[:, s: e].to(device)
            if col_x.dim() == 1:
                col_x = col_x[:, None]

            new_colx = []
            for i, row_slice_adj in enumerate(adj):
                if row_slice_adj is not None:
                    new_colx.append(matmul(row_slice_adj.to(device), col_x, reduce=reduce).to(original_device))

            new_colx = torch.cat(new_colx, dim=0)
            if inplace:
                x[:, s: e] = new_colx
            else:
                new_x[:, s: e] = new_colx

    if inplace:
        return x
    else:
        return new_x


def general_chunk_forward_beta(l: Union[torch.nn.Linear, torch.nn.LayerNorm],
                               x: torch.tensor,
                               num_chunks):
    device = x.device
    idx = get_chunk_idx(x.shape[0], num_chunks)

    if isinstance(l, torch.nn.Linear):
        inplace = l.weight.shape[0] == l.weight.shape[1]
    else:
        inplace = True

    if not inplace:
        new_x = torch.empty(x.shape[0], l.weight.shape[0], dtype=x.dtype)

    with torch.no_grad():
        pbar = tqdm(idx, desc="layer")
        for s, e in pbar:
            if inplace:
                x[s: e, :] = l(x[s: e, :].to(l.weight.device)).to(device)
            else:
                new_x[s: e, :] = l(x[s: e, :].to(l.weight.device)).to(device)

    if not inplace:
        return new_x
    else:
        return x


def chunk_element_mul_beta(x: torch.tensor, w: torch.tensor, num_chunks):
    device = x.device
    idx = get_chunk_idx(x.shape[0], num_chunks)

    with torch.no_grad():
        pbar = tqdm(idx, desc="mul_")
        for s, e in pbar:
            x[s: e, :] = (x[s: e, :].to(w.device).mul_(w)).to(device)

    return x


def chunk_add_beta(x: torch.tensor, w: torch.tensor, num_chunks):
    w = w.to(x.device)
    idx = get_chunk_idx(x.shape[0], num_chunks)

    with torch.no_grad():
        pbar = tqdm(idx, desc="add_")
        for s, e in pbar:
            x[s: e, :] = x[s: e, :].add_(w)

    return x


def chunk_nonparam_layer(x: torch.tensor, l, num_chunks):
    idx = get_chunk_idx(x.shape[0], num_chunks)

    with torch.no_grad():
        pbar = tqdm(idx, desc="nonparam layer")
        for s, e in pbar:
            x[s: e, :] = l(x[s: e, :])

    return x
