from typing import Union, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_sparse import set_diag, SparseTensor

from .chunk_func import chunked_sp_matmul, general_chunk_forward, chunk_element_mul


class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self,
                x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                size: Size = None,
                return_attention_weights=None):
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def chunked_pass(self, x, adj, num_chunks):
        hid_size = x[0].shape[-1] if isinstance(x, tuple) else x.shape[-1]
        num_chunks = min(num_chunks, hid_size, adj.sizes()[0])
        H, C = self.heads, self.out_channels

        x_l = None
        x_r = None
        alpha_l = None
        alpha_r = None
        if isinstance(x, torch.Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = general_chunk_forward(self.lin_l, x, num_chunks).view(-1, H, C)
            alpha_l = chunk_element_mul(x_l, self.att_l, num_chunks).sum(dim=-1)
            alpha_r = chunk_element_mul(x_r, self.att_r, num_chunks).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = general_chunk_forward(self.lin_l, x_l, num_chunks).view(-1, H, C)
            alpha_l = chunk_element_mul(x_l, self.att_l, num_chunks).sum(dim=-1)
            if x_r is not None:
                x_r = general_chunk_forward(self.lin_r, x_r, num_chunks).view(-1, H, C)
                alpha_r = chunk_element_mul(x_r, self.att_r, num_chunks).sum(dim=-1)

        size_chunk = x_l.shape[0] // num_chunks + (x_l.shape[0] % num_chunks > 0)
        dst_num = x_r.shape[0] // size_chunk + (x_r.shape[0] % size_chunk > 0)

        assert x_l is not None
        assert alpha_l is not None

        rows, cols, _ = adj.coo()
        att_full_mat = alpha_r[rows] + alpha_l[cols]
        att_full_mat = F.leaky_relu(att_full_mat, self.negative_slope)
        att_full_mat = softmax(att_full_mat, rows)
        att_full_mat = F.dropout(att_full_mat, p=self.dropout, training=self.training)

        x_out = []
        for i in range(H):
            att_adj = SparseTensor(row=rows,
                                   col=cols,
                                   value=att_full_mat[:, i],
                                   sparse_sizes=adj.sizes())
            x_out.append(chunked_sp_matmul(att_adj, x_l[:, i, :], num_chunks, device=self.lin_l.weight.device))

        if self.concat:
            x_out = torch.cat(x_out, dim=-1)
        else:
            x_out = sum(x_out) / len(x_out)

        if self.bias is not None:
            x_out += self.bias.cpu()

        return x_out


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.p_list = []

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads, add_self_loops=False))
        self.p_list.append({'params': self.convs[-1].parameters()})

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads, add_self_loops=False))
            self.p_list.append({'params': self.convs[-1].parameters()})

        self.convs.append(
            GATConv(heads * hidden_channels, out_channels, heads, concat=False, add_self_loops=False))
        self.p_list.append({'params': self.convs[-1].parameters(), 'weight_decay': 0.})

        self.norms = torch.nn.ModuleList()

        for _ in range(num_layers - 1):
            self.norms.append(torch.nn.LayerNorm(heads * hidden_channels, elementwise_affine=True))
            self.p_list.append({'params': self.norms[-1].parameters(), 'weight_decay': 0.})

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, data):
        x, adjs, prime_index = data.x, data.edge_index, data.output_node_mask

        if isinstance(x, torch.Tensor) and isinstance(adjs, SparseTensor):

            for i in range(self.num_layers):
                if i != self.num_layers - 1:
                    x = self.convs[i](x, adjs)
                    x = self.norms[i](x)
                    x = F.elu(x)
                    x = F.dropout(x, p=0.5, training=self.training)
                else:
                    if prime_index is not None:
                        x = self.convs[i]((x, x[prime_index]), adjs[prime_index, :])
                    else:
                        x = self.convs[i](x, adjs)

        elif isinstance(x, torch.Tensor) and isinstance(adjs, list):
            for i in range(self.num_layers):
                adj = adjs.pop(0)
                x_target = x[:adj.sizes()[0]]
                x = self.convs[i]((x, x_target), adj)

                if i != self.num_layers - 1:
                    x = self.norms[i](x)
                    x = F.elu(x)
                    x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)

    def chunked_pass(self, data, num_chunks):
        x, adjs, prime_index = data.x, data.adj, data.idx

        if isinstance(x, torch.Tensor) and isinstance(adjs, SparseTensor):

            for i in range(self.num_layers):
                if i != self.num_layers - 1:
                    x = self.convs[i].chunked_pass(x, adjs, num_chunks)
                    x = general_chunk_forward(self.norms[i], x, num_chunks)
                    x = F.elu(x)
                    x = F.dropout(x, p=0.5, training=self.training)
                else:
                    if prime_index is not None:
                        x = self.convs[i].chunked_pass((x, x[prime_index]), adjs[prime_index, :], num_chunks)
                    else:
                        x = self.convs[i].chunked_pass(x, adjs, num_chunks)

        elif isinstance(x, torch.Tensor) and isinstance(adjs, list):
            for i in range(self.num_layers):
                adj = adjs.pop(0)
                x_target = x[:adj.sizes()[0]]
                x = self.convs[i].chunked_pass((x, x_target), adj, num_chunks)

                if i != self.num_layers - 1:
                    x = general_chunk_forward(self.norms[i], x, num_chunks)
                    x = F.elu(x)
                    x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)
