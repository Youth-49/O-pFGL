import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from itertools import product
from torch_geometric.nn import SGConv
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import GCN, GIN, MLP, GraphSAGE, PNA
from torch_geometric.nn import LabelPropagation, APPNP

class MyGCN(GCN):
    def forward(self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        layer = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size = None,
        num_sampled_nodes_per_hop = None,
        num_sampled_edges_per_hop = None,):

        if (num_sampled_nodes_per_hop is not None
                and isinstance(edge_weight, Tensor)
                and isinstance(edge_attr, Tensor)):
            raise NotImplementedError("'trim_to_layer' functionality does not "
                                      "yet support trimming of both "
                                      "'edge_weight' and 'edge_attr'")

        xs = []
        assert len(self.convs) == len(self.norms)
        internal_repr = None
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if layer is not None and i == layer:
                internal_repr = x

            if (not torch.jit.is_scripting()
                    and num_sampled_nodes_per_hop is not None):
                x, edge_index, value = self._trim(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = conv(x, edge_index, edge_weight=edge_weight,
                         edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = conv(x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)

            if i < self.num_layers - 1 or self.jk_mode is not None:
                if self.act is not None and self.act_first:
                    x = self.act(x)
                if self.supports_norm_batch:
                    x = norm(x, batch, batch_size)
                else:
                    x = norm(x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                x = self.dropout(x)
                if hasattr(self, 'jk'):
                    xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        
        if layer is None:
            return x
        else:
            return internal_repr, x


from torch_geometric.utils import one_hot
class MyLabelProp(LabelPropagation):
    @torch.no_grad()
    def forward(
        self,
        y,
        edge_index,
        mask = None,
        num_classes = None,
        edge_weight = None,
        post_step = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            y (torch.Tensor): The ground-truth label information
                :math:`\mathbf{Y}`.
            edge_index (torch.Tensor or SparseTensor): The edge connectivity.
            mask (torch.Tensor, optional): A mask or index tensor denoting
                which nodes are used for label propagation.
                (default: :obj:`None`)
            edge_weight (torch.Tensor, optional): The edge weights.
                (default: :obj:`None`)
            post_step (callable, optional): A post step function specified
                to apply after label propagation. If no post step function
                is specified, the output will be clamped between 0 and 1.
                (default: :obj:`None`)
        """
        if y.dtype == torch.long and y.size(0) == y.numel():
            y = one_hot(y.view(-1), num_classes)

        out = y
        if mask is not None:
            out = torch.zeros_like(y)
            out[mask] = y[mask]

        if isinstance(edge_index, SparseTensor) and not edge_index.has_value():
            edge_index = gcn_norm(edge_index, add_self_loops=False)
        elif isinstance(edge_index, Tensor) and edge_weight is None:
            edge_index, edge_weight = gcn_norm(edge_index, num_nodes=y.size(0),
                                               add_self_loops=False)

        res = (1 - self.alpha) * out
        for _ in range(self.num_layers):
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight)
            out.mul_(self.alpha).add_(res)
            if post_step is not None:
                out = post_step(out)
            else:
                out.clamp_(0., 1.)

        return out


class SIGN(MessagePassing):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, K=2, dropout=0, add_self_loops: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        # super(SIGN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.dropout = dropout
        self.K = K
        self.add_self_loops = add_self_loops
        self.feat_list_dim = (self.K + 1) * in_channels
        if self.num_layers == 1:
            self.mlp = torch.nn.ModuleList()
            self.mlp.append(nn.Linear(self.feat_list_dim, out_channels))
            self.mlp = nn.Sequential(*self.mlp)
        else:
            self.mlp = torch.nn.ModuleList()
            self.mlp.append(nn.Linear(self.feat_list_dim, hidden_channels))
            self.mlp.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(self.num_layers - 2):
                self.mlp.append(nn.Linear(hidden_channels, hidden_channels))
                self.mlp.append(nn.BatchNorm1d(hidden_channels))
            self.mlp.append(nn.Linear(hidden_channels, out_channels))
            self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        feat_list = [x]
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)

        for k in range(self.K):
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            feat_list.append(x)

        if self.K >= 1:
            feat_list = torch.cat(feat_list, dim=1)
        else:
            feat_list = feat_list[0]

        output = self.mlp(feat_list)
        return output
    
    def reset_parameters(self):
        for layer in self.mlp:
            layer.reset_parameters()


class SSGC(MessagePassing):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, K=2, dropout=0, add_self_loops: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        # super(SIGN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.dropout = dropout
        self.K = K
        self.add_self_loops = add_self_loops
        self.feat_list_dim = in_channels
        if self.num_layers == 1:
            self.mlp = torch.nn.ModuleList()
            self.mlp.append(nn.Linear(self.feat_list_dim, out_channels))
            self.mlp = nn.Sequential(*self.mlp)
        else:
            self.mlp = torch.nn.ModuleList()
            self.mlp.append(nn.Linear(self.feat_list_dim, hidden_channels))
            self.mlp.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(self.num_layers - 2):
                self.mlp.append(nn.Linear(hidden_channels, hidden_channels))
                self.mlp.append(nn.BatchNorm1d(hidden_channels))
            self.mlp.append(nn.Linear(hidden_channels, out_channels))
            self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        feat_list = [x]
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)

        for k in range(self.K):
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            feat_list.append(x)

        if self.K >= 1:
            feat_list = torch.cat(feat_list, dim=1)
            feat_list = torch.mean(feat_list, dim=1)
        else:
            feat_list = feat_list[0]

        output = self.mlp(feat_list)
        return output
    
    def reset_parameters(self):
        for layer in self.mlp:
            layer.reset_parameters()


class MyAPPNP(MessagePassing):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, K=2, dropout=0, alpha=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.dropout = dropout
        self.K = K
        self.alpha = alpha
        self.appnp = APPNP(K=K, alpha=alpha)
        if self.num_layers == 1:
            self.mlp = torch.nn.ModuleList()
            self.mlp.append(nn.Linear(self.in_channels, out_channels))
            self.mlp = nn.Sequential(*self.mlp)
        else:
            self.mlp = torch.nn.ModuleList()
            self.mlp.append(nn.Linear(self.in_channels, hidden_channels))
            self.mlp.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(self.num_layers - 2):
                self.mlp.append(nn.Linear(hidden_channels, hidden_channels))
                self.mlp.append(nn.BatchNorm1d(hidden_channels))
            self.mlp.append(nn.Linear(hidden_channels, out_channels))
            self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        x = self.appnp(x, edge_index, edge_weight)
        output = self.mlp(x)
        return output
    
    def reset_parameters(self):
        for layer in self.mlp:
            layer.reset_parameters()


class PGE(nn.Module):

    def __init__(self, nfeat, nnodes, nhid=128, nlayers=3, device=None, args=None):
        super(PGE, self).__init__()
        if args.dataset in ['ogbn-arxiv', 'arxiv', 'flickr']:
           nhid = 256
        if args.dataset in ['reddit']:
           nhid = 256
           if args.rate==0.01:
               nhid = 128
           nlayers = 3
           # nhid = 128

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(nfeat*2, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhid))
        for i in range(nlayers-2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.layers.append(nn.Linear(nhid, 1))

        edge_index = np.array(list(product(range(nnodes), range(nnodes))))
        self.edge_index = edge_index.T
        self.nnodes = nnodes
        self.device = device
        self.reset_parameters()
        self.cnt = 0
        self.args = args
        self.nnodes = nnodes

    def forward(self, x, inference=False):
        if self.args.dataset == 'ogbn-products':
            edge_index = self.edge_index
            n_part = 5
            splits = np.array_split(np.arange(edge_index.shape[1]), n_part)
            edge_embed = []
            for idx in splits:
                tmp_edge_embed = torch.cat([x[edge_index[0][idx]],
                        x[edge_index[1][idx]]], axis=1)
                for ix, layer in enumerate(self.layers):
                    tmp_edge_embed = layer(tmp_edge_embed)
                    if ix != len(self.layers) - 1:
                        tmp_edge_embed = self.bns[ix](tmp_edge_embed)
                        tmp_edge_embed = F.relu(tmp_edge_embed)
                edge_embed.append(tmp_edge_embed)
            edge_embed = torch.cat(edge_embed)
        else:
            edge_index = self.edge_index
            edge_embed = torch.cat([x[edge_index[0]],
                    x[edge_index[1]]], axis=1)
            for ix, layer in enumerate(self.layers):
                edge_embed = layer(edge_embed)
                if ix != len(self.layers) - 1:
                    edge_embed = self.bns[ix](edge_embed)
                    edge_embed = F.relu(edge_embed)

        adj = edge_embed.reshape(self.nnodes, self.nnodes)

        adj = (adj + adj.T)/2
        adj = torch.sigmoid(adj)
        adj = adj - torch.diag(torch.diag(adj, 0))
        return adj

    @torch.no_grad()
    def inference(self, x):
        # self.eval()
        adj_syn = self.forward(x, inference=True)
        return adj_syn

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)


class FedTAD_ConGenerator(nn.Module):

    def __init__(self, noise_dim, feat_dim, out_dim, dropout):
        super(FedTAD_ConGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.emb_layer = nn.Embedding(out_dim, out_dim)
        
        hid_layers = []
        dims = [noise_dim+out_dim, 64, 128, 256]
        for i in range(len(dims)-1):
            d_in = dims[i]
            d_out = dims[i+1]
            hid_layers.append(nn.Linear(d_in, d_out))
            hid_layers.append(nn.Tanh())
            hid_layers.append(nn.Dropout(p=dropout, inplace=False))
        self.hid_layers = nn.Sequential(*hid_layers)
        self.nodes_layer = nn.Linear(256, feat_dim)

    def forward(self, z, c):
        z_c = torch.cat((self.emb_layer.forward(c), z), dim=-1)
        hid = self.hid_layers(z_c)
        node_logits = self.nodes_layer(hid)
        return node_logits

# For FedSpray
class MLPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        output = torch.mm(x, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.l1 = MLPLayer(in_channel, out_channel)

    def forward(self, x, proxy=None):
        x = self.l1(x)
        x = F.relu(x)
        return x

class Classifier(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.l2 = MLPLayer(in_channel, out_channel)

    def forward(self, x, proxy=None):
        if proxy is not None:
            x = x + proxy
        #     x1 = torch.concat([x1, proxy], dim=1)
        x = self.l2(x)
        return x