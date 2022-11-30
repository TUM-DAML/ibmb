from torch_geometric.nn import GCNConv
import torch


class Vanilla_GCN(torch.nn.Module):
    
    def __init__(self, in_size, hid_size, num_classes, num_layers):
        super(Vanilla_GCN, self).__init__()
        
        size_list = [in_size] + [hid_size] * num_layers + [num_classes]
        self.layers = torch.nn.ModuleList([])
        for i in range(len(size_list) - 2):
            self.layers.append(GCNConv(in_channels=size_list[i], out_channels=size_list[i+1], add_self_loops=False))
            self.layers.append(torch.nn.LayerNorm(size_list[i+1], elementwise_affine=True))
            self.layers.append(torch.nn.ReLU(inplace=True))
            self.layers.append(torch.nn.Dropout(p=0.5))
            
        self.layers.append(torch.nn.Linear(size_list[-2], size_list[-1]))
        
        self.p_list = []
        for l in self.layers:
            if isinstance(l, (torch.nn.Linear, GCNConv)):
                self.p_list.append({'params': l.parameters()})
            elif isinstance(l, torch.nn.LayerNorm):
                self.p_list.append({'params': l.parameters(), 'weighted_decay': 0.})
                
    
    def forward(self, graph):
        x = graph.x
        try:
            edge_index = graph.adj_t
        except:
            edge_index = graph.edge_index
        
        for l in self.layers:
            if isinstance(l, GCNConv):
                x = l(x, edge_index)
            else:
                x = l(x)
        
        return x.log_softmax(dim=-1)