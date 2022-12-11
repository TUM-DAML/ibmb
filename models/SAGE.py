import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.typing import OptPairTensor
from torch_sparse import SparseTensor, matmul

from .chunk_func import chunked_sp_matmul, general_chunk_forward


class MySAGEConv(SAGEConv):
    def forward(self, x, adj_t, prime_index=None, size=None):
        if isinstance(x, torch.Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        if adj_t.has_value():
            adj_t = adj_t.set_value(None, layout=None)
        out = matmul(adj_t, x[0], reduce=self.aggr)  # mean
        out = self.lin_l(out)

        x_r = x[1]
        # print(f'x0: {x[0].shape}, adj: {adj_t.sizes()}, x_l: {out.shape}, x_r: {x_r.shape}\n')
        if self.root_weight and x_r is not None:
            if prime_index is not None:
                x_r = x_r[prime_index, :]
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def chunked_pass(self, x, adj_t, num_chunks, prime_index=None, size=None):
        if isinstance(x, torch.Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        if adj_t.has_value():
            adj_t = adj_t.set_value(None, layout=None)
        out = chunked_sp_matmul(adj_t, x[0], num_chunks, reduce=self.aggr, device=self.lin_l.weight.device)
        out = general_chunk_forward(self.lin_l, out, num_chunks)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            if prime_index is not None:
                x_r = x_r[prime_index, :]
            out += general_chunk_forward(self.lin_r, x_r, num_chunks)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


class SAGEModel(torch.nn.Module):
    def __init__(self,
                 num_node_features,
                 num_classes,
                 hidden_channels,
                 num_layers):
        super(SAGEModel, self).__init__()

        #         torch.manual_seed(2021)
        #         torch.cuda.manual_seed(2021)

        self.layers = torch.nn.ModuleList([])
        self.p_list = []

        for i in range(num_layers):
            in_channels = num_node_features if i == 0 else hidden_channels
            self.layers.append(MySAGEConv(in_channels=in_channels, out_channels=hidden_channels))
            self.p_list.append({'params': self.layers[-1].parameters()})

            self.layers.append(torch.nn.LayerNorm(hidden_channels, elementwise_affine=True))
            self.p_list.append({'params': self.layers[-1].parameters(), 'weighted_decay': 0.})

            self.layers.append(torch.nn.ReLU(inplace=True))
            self.layers.append(torch.nn.Dropout(p=0.5))

        self.layers.append(torch.nn.Linear(in_features=hidden_channels, out_features=num_classes))
        self.p_list.append({'params': self.layers[-1].parameters(), 'weighted_decay': 0.})

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()

    def forward(self, data):
        x, adjs, prime_index = data.x, data.edge_index, data.output_node_mask

        if isinstance(adjs, SparseTensor):

            for i, l in enumerate(self.layers):
                if isinstance(l, MySAGEConv):
                    if i == len(self.layers) - 5 and prime_index is not None:
                        x = l(x, adjs[prime_index, :], prime_index)
                    else:
                        x = l(x, adjs)
                else:
                    x = l(x)

        elif isinstance(adjs, list):

            for i, l in enumerate(self.layers):
                if isinstance(l, MySAGEConv):
                    a = adjs.pop(0)
                    x_target = x[:a.sizes()[0], :]
                    # print(f'a:{a.sizes()}, x:{x.shape}, x_tar:{x_target.shape}')
                    x = l((x, x_target,), a)
                else:
                    x = l(x)

        return x.log_softmax(dim=-1)

    def chunked_pass(self, data, num_chunks):
        x, adjs, prime_index = data.x, data.adj, data.idx

        assert isinstance(adjs, SparseTensor)

        for i, l in enumerate(self.layers):
            if isinstance(l, MySAGEConv):
                if i == len(self.layers) - 5 and prime_index is not None:
                    x = l.chunked_pass(x, adjs[prime_index, :], num_chunks, prime_index=prime_index)
                else:
                    x = l.chunked_pass(x, adjs, num_chunks, )
            elif isinstance(l, (torch.nn.Linear, torch.nn.LayerNorm)):
                x = general_chunk_forward(l, x, num_chunks)
            else:  # relu, dropout
                x = l(x)

        return x.log_softmax(dim=-1)
