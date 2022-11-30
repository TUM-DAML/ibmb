import numpy as np
import torch
import math
from tqdm import tqdm
from data import normalize_adjmat


def ppr_power_method(W, batch, topk, num_iter, alpha):
    topk_neighbors = []
    logits = torch.zeros(W.size(0), len(batch), device=W.device())
    for i, tele_set in enumerate(batch):
        logits[tele_set, i] = 1 / len(tele_set)

    new_logits = logits.clone()
    for i in range(num_iter):
        new_logits = W @ new_logits * (1 - alpha) + alpha * logits

    inds = new_logits.argsort(0)
    nonzeros = (new_logits > 0).sum(0)
    nonzeros = torch.minimum(nonzeros, torch.tensor([topk], dtype=torch.int64, device=W.device()))
    for i in range(new_logits.shape[1]):
        topk_neighbors.append(inds[-nonzeros[i]:, i].cpu().detach().numpy())
    
    return topk_neighbors


def hk_power_method(W, batch, topk, num_iter, temp):
    topk_neighbors = []
    logits = torch.zeros(W.size(0), len(batch), device=W.device())
    for i, tele_set in enumerate(batch):
        logits[tele_set, i] = 1 / len(tele_set)

    coeff = math.e ** -temp
    sum_logits = logits.clone() * coeff
    
    for i in range(1, num_iter + 1):
        coeff *= temp / i
        logits = W @ logits
        sum_logits += logits * coeff

    inds = sum_logits.argsort(0)
    nonzeros = (sum_logits > 0).sum(0)
    nonzeros = torch.minimum(nonzeros, torch.tensor([topk], dtype=torch.int64, device=W.device()))
    for i in range(sum_logits.shape[1]):
        topk_neighbors.append(inds[-nonzeros[i]:, i].cpu().detach().numpy())
    
    return topk_neighbors


def partition_fixed_loader(neighbor_sampling, 
                            adj, 
                            num_nodes, 
                            partitions, 
                            prime_indices, 
                            topk, 
                            power_batch_size=50, 
                            partition_diffusion_param=None, 
                            expand=False):
    
    # force rw matrix
    mat = normalize_adjmat(adj, normalization = 'rw')
    mat = mat.cuda()
    
    nodes = []
    loader = []
    
    if neighbor_sampling == 'batch_ppr':
        power_method = ppr_power_method
        if partition_diffusion_param is None:
            partition_diffusion_param = 0.2
        
    elif neighbor_sampling == 'batch_hk':
        power_method = hk_power_method
        if partition_diffusion_param is None:
            partition_diffusion_param = 5
            
    else:
        raise NotImplementedError
    
    for n, part in enumerate(tqdm(partitions)):
        primes_in_part = np.intersect1d(part, prime_indices)
        if len(primes_in_part):
            nodes.append(primes_in_part)
            
        if len(nodes) >= power_batch_size or n == len(partitions) - 1:
            topk_neighbors = power_method(mat, nodes, topk, 50, partition_diffusion_param)
            for i in range(len(nodes)):
                seconds = np.union1d(nodes[i], topk_neighbors[i])
                if expand:
                    primes_in_part = np.intersect1d(seconds, prime_indices)
                    loader.append((primes_in_part, seconds))
                else:
                    loader.append((nodes[i], seconds))
            nodes = []
        torch.cuda.empty_cache()
    
    mat = None
    torch.cuda.reset_peak_memory_stats()
    return loader
