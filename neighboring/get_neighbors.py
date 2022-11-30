import logging

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from .pernode_hk_neighbor import topk_hk_neighbors


def get_neighbors(mode: str, 
                  neighbor_sampling: str, 
                  prime_indices: np.ndarray, 
                  scipy_adj: csr_matrix, 
                  ppr_mat: csr_matrix, 
                  topk: int = 64) -> list:
    
    neighbors = None
    
    if mode in ['ppr', 'rand', 'randfix'] or 'ppr' in [mode, neighbor_sampling]:
        if neighbor_sampling == 'ppr':
            
            neighbors = []
            lens = []
            assert ppr_mat is not None      

            for i, n in enumerate(tqdm(prime_indices)):
        
                # choice 1, if already contains top-k
                nodes = ppr_mat.indices[ppr_mat.indptr[i] : ppr_mat.indptr[i + 1]]

                # choice 2, sort from a denser matrix
#                 row = ppr_mat.getrow(i)
#                 ind, vals = find(row)[1:]
#                 mask = np.argpartition(vals, kth = max(0, len(vals) - topk))[-topk:]
#                 nodes = ind[mask]

                nodes = np.union1d(nodes, n)
                neighbors.append(nodes.astype(np.int64))
                lens.append(len(nodes))
            
            logging.info(f'mean num neighbors: {sum(lens) / len(lens)}')

        elif neighbor_sampling == 'pnorm':
            raise NotImplementedError
    #             train_neighbors = separate_pnorm_neighbors()
    
        elif neighbor_sampling == 'hk':
            neighbors = topk_hk_neighbors(scipy_adj, prime_indices, topk=topk)
        
    return neighbors
