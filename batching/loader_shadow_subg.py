from typing import List
from tqdm import tqdm
import numpy as np
import numba
import torch
from torch_sparse import SparseTensor
# from torch_geometric.utils import subgraph
from torch_geometric.data import Data
import pdb


@numba.njit(cache=True)
def subgraph(full_edge_index: np.ndarray, full_edge_weight: np.ndarray, nnodes: int, neighbors: np.ndarray):
    n_mask = np.zeros(nnodes, dtype=np.bool_)
    n_idx = np.zeros(nnodes, dtype=np.int64)
    
    for i, n in enumerate(neighbors):
        n_mask[n] = True
        n_idx[n] = i
    
    mask = np.logical_and(n_mask[full_edge_index[0]], n_mask[full_edge_index[1]])
    edge_index = full_edge_index[:, mask]
    edge_attr = full_edge_weight[mask]
    for col in range(edge_index.shape[1]):
        n1, n2 = edge_index[:, col]
        edge_index[0, col] = n_idx[n1]
        edge_index[1, col] = n_idx[n2]
    
    return edge_index, edge_attr


def shodaow_subgraph_loader(neighbors: List[np.ndarray], prime_indices: np.ndarray, adj: SparseTensor):
    """
    Args:
        neighbors: List[np.ndarray] containing topk neighbors for each primary node
        prime_indices: np.ndarray
        adj: int, expected number of batches
    
    Returns:
        List[adj]
    """
    full_edge_index = torch.vstack([adj.storage._row, adj.storage._col]).numpy()
    full_edge_weight = adj.storage._value.numpy()
    nnodes = adj.sizes()[0]
    batches = []
    
    for i, node in enumerate(tqdm(prime_indices)):
        cur_neighbors = neighbors[i]
                
        edge_index, edge_attr = subgraph(full_edge_index, full_edge_weight, nnodes, cur_neighbors)
        seed_node_mask = cur_neighbors == node
                
        batches.append(Data(edge_index=torch.from_numpy(edge_index), 
                            edge_attr=torch.from_numpy(edge_attr), 
                            seed_in_batch_mask=torch.from_numpy(seed_node_mask), 
                            seed_idx=torch.tensor([node]),
                            batch_node_idx=torch.from_numpy(cur_neighbors)))
            
    return batches