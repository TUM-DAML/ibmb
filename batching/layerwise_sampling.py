import numpy as np


def ladies_sampler(batch_nodes, samp_num_list, mat):

    previous_nodes = batch_nodes
    adjs  = []
    node_indices = [batch_nodes]

    for num in samp_num_list:
        U = mat[previous_nodes , :]
        
        pi = np.sum(U.power(2), axis=0).A1
        nonzero_mask = np.where(pi > 0)[0]
        p = pi[nonzero_mask]
        p /= p.sum()
        
        s_num = min(len(nonzero_mask), num)
        after_nodes = np.random.choice(nonzero_mask, s_num, p = p, replace = False)
        after_nodes = np.union1d(after_nodes, batch_nodes)
        adj = U[: , after_nodes]
        adjs.append((adj.indptr.astype(np.int64), adj.indices.astype(np.int64), adj.data, adj.shape))
        node_indices.append(after_nodes)
        previous_nodes = after_nodes
        
    adjs.reverse()
    node_indices.reverse()
    
    return adjs, node_indices


class LadiesLoader():
    def __init__(self, prime_nodes, samp_num_list, mat, num_parts = None):
        np.random.seed(2021)
        self.fix_batch_prime = isinstance(prime_nodes, list)
        self.prime_nodes = prime_nodes
        self.num_parts = num_parts
        self.samp_num_list = samp_num_list
        self.mat = mat
        
        self.reset()
        
        
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
    
    def __len__(self):
        if self.fix_batch_prime:
            return len(self.prime_nodes)
        else:
            return self.num_parts
        
    def reset(self):
        if not self.fix_batch_prime:
            if isinstance(self.prime_nodes, list):
                self.prime_nodes = np.concatenate(self.prime_nodes)
            self.prime_nodes = np.array_split(np.random.permutation(self.prime_nodes), self.num_parts)
        
        self.fetch_idx = list(np.random.permutation(len(self.prime_nodes)))
        
    def next(self):
        if not len(self.fetch_idx):
            self.reset()
            raise StopIteration()
            
        i = self.fetch_idx.pop(0)
        batch_nodes = self.prime_nodes[i]
        adjs, node_indices = ladies_sampler(batch_nodes, self.samp_num_list, self.mat)
        return (len(self.fetch_idx) == 0, (adjs, node_indices))
        