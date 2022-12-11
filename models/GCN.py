import torch
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor, matmul

from .chunk_func import chunked_sp_matmul, general_chunk_forward


class MyGCNConv(GCNConv):
    def forward(self, x, edge_index):
        x = self.lin(x)
        out = matmul(edge_index, x, reduce=self.aggr)
        return out
    
    def chunked_pass(self, x, edge_index, num_chunks):
        x = general_chunk_forward(self.lin, x, num_chunks)
        x = chunked_sp_matmul(edge_index, x, num_chunks, reduce=self.aggr, device=x.device)
        return x
        

class GCN(torch.nn.Module):
    def __init__(self, 
                 num_node_features, 
                 num_classes, 
                 hidden_channels, 
                 num_layers):
        super(GCN, self).__init__()
    
#         torch.manual_seed(2021)
#         torch.cuda.manual_seed(2021)

        self.layers = torch.nn.ModuleList([])
        self.p_list = []
        
        for i in range(num_layers):
            in_channels = num_node_features if i == 0 else hidden_channels
            self.layers.append(MyGCNConv(in_channels=in_channels,  out_channels=hidden_channels))
            self.p_list.append({'params': self.layers[-1].parameters()})
            
            self.layers.append(torch.nn.LayerNorm(hidden_channels, elementwise_affine=True))
            self.p_list.append({'params': self.layers[-1].parameters(), 'weighted_decay': 0.})
            
            self.layers.append(torch.nn.ReLU(inplace=True))
            self.layers.append(torch.nn.Dropout(p=0.5))

        self.layers.append(torch.nn.Linear(hidden_channels, num_classes))
        self.p_list.append({'params': self.layers[-1].parameters(), 'weight_decay': 0.})

    def forward(self, data):
        x, adjs, prime_index = data.x, data.edge_index, data.output_node_mask
        
        if isinstance(adjs, SparseTensor):

            for i, l in enumerate(self.layers):
                if isinstance(l, MyGCNConv):
                    if i == len(self.layers) - 5 and prime_index is not None:
                        x = l(x, adjs[prime_index, :])
                    else:
                        x = l(x, adjs)
                else:
                    x = l(x)
                    
        elif isinstance(adjs, list):
            
            for i, l in enumerate(self.layers):
                if isinstance(l, MyGCNConv):
                    x = l(x, adjs.pop(0))
                else:
                    x = l(x)

        return x.log_softmax(dim=-1)
    
    def chunked_pass(self, data, num_chunks):
        x, adjs, prime_index = data.x, data.adj, data.idx
        
        assert isinstance(adjs, SparseTensor)

        for i, l in enumerate(self.layers):
            if isinstance(l, MyGCNConv):
                if i == len(self.layers) - 5 and prime_index is not None:
                    x = l.chunked_pass(x, adjs[prime_index, :], num_chunks)
                else:
                    x = l.chunked_pass(x, adjs, num_chunks)
            elif isinstance(l, (torch.nn.Linear, torch.nn.LayerNorm)):
                x = general_chunk_forward(l, x, num_chunks)
            else:   # relu, dropout
                x = l(x)

        return x.log_softmax(dim=-1)
    
    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()
