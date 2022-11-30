from torch_sparse import SparseTensor, matmul
import torch


def chunked_matmul(w: torch.tensor, x: torch.tensor, num_chunks: int):
    """
    x @ w -> [x1; x2; x3; ...] @ w
    """
    device = x.device
    size_chunks = x.shape[0] // num_chunks + ((x.shape[0] % num_chunks) > 0)
    x_chunks = []
    for i in range(num_chunks):
        x_chunks.append((x[i * size_chunks : (i + 1) * size_chunks, :].to(w.device) @ w).to(device))
    
    return torch.cat(x_chunks, dim=0)


def chunked_sp_matmul(adj: SparseTensor, 
                      x: torch.tensor, 
                      num_chunks: int, 
                      reduce: str = 'add', 
                      device: torch.device = torch.device('cuda')):
    """
    adj @ x -> [adj1; adj2; adj3; ...] @ [x1, x2, x3, ...]
    """
    num_chunks = min(num_chunks, x.shape[-1])
    original_device = x.device
    size_chunks_0 = adj.sizes()[0] // num_chunks + ((adj.sizes()[0] % num_chunks) > 0)
    size_chunks_1 = x.shape[1] // num_chunks + ((x.shape[1] % num_chunks) > 0)
    
    m_chunks_0 = adj.sizes()[0] // size_chunks_0 + ((adj.sizes()[0] % size_chunks_0) > 0)
    m_chunks_1 = x.shape[1] // size_chunks_1 + ((x.shape[1] % size_chunks_1) > 0)
    
    new_x = []
    
    for i in range(m_chunks_0):
        row_slice_adj = adj[i * size_chunks_0 : min((i + 1) * size_chunks_0, adj.sizes()[0]), :].to(device)
        row_x = []
        for j in range(m_chunks_1):
            col_x = x[:, j * size_chunks_1 : (j + 1) * size_chunks_1].to(device)
            if col_x.dim() == 1:
                col_x = col_x[:, None]
            row_x.append( matmul(row_slice_adj, col_x, reduce=reduce).to(original_device) )
        
        new_x.append(torch.cat(row_x, dim=1))
    
    new_x = torch.cat(new_x, dim=0)
    return new_x


def general_chunk_forward(l, x, num_chunks):
    device = x.device
    size_chunks = x.shape[0] // num_chunks + ((x.shape[0] % num_chunks) > 0)
    new_x = []
    for i in range(num_chunks):
        new_x.append(l(x[i * size_chunks : (i + 1) * size_chunks, :].to(l.weight.device)).to(device))
    
    return torch.cat(new_x, dim=0)


def chunk_element_mul(x, w, num_chunks):
    device = x.device
    new_x = []
    size_chunks = x.shape[0] // num_chunks + ((x.shape[0] % num_chunks) > 0)
    for i in range(num_chunks):
        new_x.append((x[i * size_chunks : (i + 1) * size_chunks, :].to(w.device) * w).to(device))
    return torch.cat(new_x, dim=0)
