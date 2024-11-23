import argparse
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from torch.optim import Adam
import torch.nn.functional as F
from util.task_util import accuracy, f1_score
from util.base_util import seed_everything, load_dataset
from deeprobust.graph.utils import is_sparse_tensor, normalize_adj_tensor
from torch_sparse import SparseTensor, fill_diag, mul
from torch_sparse import sum as sparsesum
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from model import PGE, SIGN, SSGC, MyAPPNP
from torch_geometric.nn.models import GCN
from torch_geometric.nn import SGConv
from model import MyGCN, MyLabelProp
import math
import copy
import random
import logging
import pickle
import sys
import os

# warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

# experimental environment setup
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--method', type=str, default='homo')
parser.add_argument('--personalized', action='store_true')
parser.add_argument('--root', type=str, default='./dataset')
parser.add_argument('--save_root_path', type=str, default='./main_results')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--dataset', type=str, default="Cora")
parser.add_argument('--partition', type=str, default="Louvain+", choices=["Louvain+", "Metis+"])
parser.add_argument('--part_delta', type=int, default=20)
parser.add_argument('--louvain_num_coms', type=int, default=100)
parser.add_argument('--metis_num_coms', type=int, default=100)
parser.add_argument('--num_clients', type=int, default=10)
parser.add_argument('--num_rounds', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--frac', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--hid_dim', type=int, default=256)
parser.add_argument('--nlayers', type=int, default=2)

parser.add_argument('--hops', type=int, default=2)
parser.add_argument('--rate', type=float, default=0.026)
parser.add_argument('--ipc', type=int, default=1)
parser.add_argument('--K', type=int, default=2)
parser.add_argument('--lr_feat', type=float, default=5e-3)
parser.add_argument('--lr_adj', type=float, default=1e-3)
parser.add_argument('--lr_validation_model', type=float, default=1e-4)
parser.add_argument('--lr_ft', type=float, default=1e-4)
parser.add_argument('--threshold', type=float, default=0.01, help='adj threshold.')
parser.add_argument('--feat_alpha', type=float, default=10, help='feat loss term.')
parser.add_argument('--smoothness_alpha', type=float, default=0.1, help='smoothness loss term.')
parser.add_argument('--condensing_loop', type=int, default=3000)
parser.add_argument('--validation_loop', type=int, default=3000)
parser.add_argument('--ft_loop', type=int, default=3000)
parser.add_argument('--validation_model', type=str, default="GCN")
parser.add_argument('--val_stage', type=int, default=100)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--local_strategy', type=str, default="global", choices=['global', 'ft', 'ft_imb', 'reg_ft', 'nd_lp_reg_ft'])
parser.add_argument('--reg_a', type=float, default=0.2, help='regularization coeff')
parser.add_argument('--reg_max', type=float, default=None, help='max regularization coeff')
parser.add_argument('--reg_min', type=float, default=None, help='min regularization coeff')
parser.add_argument('--lp_iter', type=int, default=2)
parser.add_argument('--lp_alpha', type=float, default=0.9)
parser.add_argument('--topk', type=int, default=0)
parser.add_argument('--thres', type=float, default=0.95)
parser.add_argument('--d_thres', type=int, default=0)

args = parser.parse_args()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))

    
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert flow in ["source_to_target"]
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        idx = col if flow == "source_to_target" else row
        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class Client:
    def __init__(self, args, client_id, model, model_name, local_data, propagate_model, num_classes, device):
        self.args = args
        self.client_id = client_id
        self.init_model = copy.deepcopy(model)
        self.model = copy.deepcopy(model)
        self.model_name = model_name
        self.num_classes = num_classes
        self.local_data = local_data
        self.propagate_model = propagate_model
        self.device = device

        self.pseudo_graph = None
        self.ensemble_model = None
        if self.args.dataset in ["Reddit2",]:
            from sklearn.preprocessing import StandardScaler
            local_feat_train = self.local_data.x[self.local_data.train_idx]
            scaler = StandardScaler()
            scaler.fit(local_feat_train.cpu().numpy())
            self.local_data.x = scaler.transform(self.local_data.x.cpu().numpy())
            self.local_data.x = torch.FloatTensor(self.local_data.x).to(self.device)

        if self.args.dataset in ['ogbn-products', 'Reddit', 'Reddit2'] and os.path.exists(f"{self.args.dataset}_{self.args.partition}_{self.args.num_clients}_{self.client_id}.pt"):
            logging.info(f"loading {self.args.dataset}_{self.args.partition}_{self.args.num_clients}_{self.client_id}.pt")
            self.local_homophily = torch.load(f"{self.args.dataset}_{self.args.partition}_{self.args.num_clients}_{self.client_id}.pt").to(self.device)
            with open(f"saving {self.args.dataset}_{self.args.partition}_{self.args.num_clients}_{self.client_id}_class.pkl", 'rb') as f:
                self.local_homophily_class = pickle.load(f)
        else:
            self.local_homophily, self.local_homophily_class, self.train_num_per_class = self.label_homophily_details()
            self.local_homophily = torch.tensor(self.local_homophily).to(self.device).unsqueeze(-1)
            if self.args.dataset in ['ogbn-products', 'Reddit', 'Reddit2']:
                logging.info(f"saving {self.args.dataset}_{self.args.partition}_{self.args.num_clients}_{self.client_id}.pt")
                torch.save(self.local_homophily.cpu(), f"{self.args.dataset}_{self.args.partition}_{self.args.num_clients}_{self.client_id}.pt")
                with open(f"saving {self.args.dataset}_{self.args.partition}_{self.args.num_clients}_{self.client_id}_class.pkl", 'wb') as f:
                    pickle.dump(self.local_homophily_class, f)

        self.train_homo_nd_coeff = (1.0 - self.local_homophily) * self.args.reg_a
        self.global_logits_class_coeff = {c: 1. / (1 + math.log(self.local_homophily_class[c] + 1)) for c in range(self.num_classes)}
        logging.info(f"client {client_id}, accumulate homo:\n{self.local_homophily_class}")
        logging.info(f"client {client_id}, class coeff from accumulate homo:\n{self.global_logits_class_coeff}")
        self.glocal_class_coeff_tensor = torch.tensor([self.global_logits_class_coeff[c] for c in range(self.num_classes)]).to(self.device).unsqueeze(-1)
        self.lp_res = self.label_prop()
        self.local_nd_coeff = self.args.reg_a * (self.lp_res @ self.glocal_class_coeff_tensor)
        if self.args.reg_max is not None or self.args.reg_min is not None:
            self.local_nd_coeff = torch.clamp(self.local_nd_coeff, min=self.args.reg_min, max=self.args.reg_max)


    def train(self):
        loss_fn = nn.CrossEntropyLoss()
        model_optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        for epoch_id in range(self.args.num_epochs):
            self.model.train()
            model_optimizer.zero_grad()
            
            logits = self.model.forward(self.local_data.x, self.local_data.edge_index)
            loss_train = loss_fn(logits[self.local_data.train_idx], self.local_data.y[self.local_data.train_idx])
            loss_train.backward()
            model_optimizer.step()
            # logging.info(f"Epoch id {epoch_id}, loss_train: {loss_train:.6f}")

        total_sample = self.local_data.x.shape[0]
        return self.model.parameters(), total_sample
    
    def get_distribution(self):
        num_nodes = self.local_data.x.shape[0]
        self.adj = sp.csr_matrix((np.ones(self.local_data.edge_index.shape[1]), 
            (self.local_data.edge_index[0].cpu(), self.local_data.edge_index[1].cpu())), shape=(num_nodes, num_nodes))

        if self.args.dataset in ["Reddit", "Flickr", "Reddit2"]:
            from util.base_data_util import mask_to_index
            idx_train = mask_to_index(self.local_data.train_idx, num_nodes)
            self.adj_train = self.adj[np.ix_(idx_train, idx_train)]
            print('size of adj_train:', self.adj_train.shape)
            print('# edges in adj_train:', self.adj_train.sum())

            feat_train = self.local_data.x[idx_train]

            if sp.issparse(self.adj_train):
                self.adj_train = sparse_mx_to_torch_sparse_tensor(self.adj_train)
            else:
                self.adj_train = torch.FloatTensor(self.adj_train)

            self.adj_train = self.adj_train.to(self.device)

            if is_sparse_tensor(self.adj_train):
                self.adj_train = normalize_adj_tensor(self.adj_train, sparse=True)
            else:
                self.adj_train = normalize_adj_tensor(self.adj_train)

            self.adj_train=SparseTensor(row=self.adj_train._indices()[0], col=self.adj_train._indices()[1],value=self.adj_train._values(), sparse_sizes=self.adj_train.size()).t()
            self.propagate_model.eval()
            feat_list = feat_train
            tmp = feat_train
            for _ in range(self.args.hops):
                aggr_x = self.propagate_model.convs[0].propagate(self.adj_train.to(self.device), x=tmp).detach()
                feat_list = torch.cat((feat_list, aggr_x), dim=1).to(self.device)
                tmp = aggr_x

            labels = self.local_data.y[self.local_data.train_idx]
            unique_labels, counts = torch.unique(labels, return_counts=True)
            unique_labels, counts = unique_labels.tolist(), counts.tolist()
            coeff = {}
            concat_feat_mean = {}
            concat_feat_std = {}
            for i, lbl in enumerate(unique_labels):
                index = torch.where(labels==lbl)
                if counts[i] > 1:
                    coeff[lbl] = counts[i]
                    concat_feat_mean[lbl] = feat_list[index].mean(dim=0).to(self.device)
                    concat_feat_std[lbl] = feat_list[index].std(dim=0).to(self.device)

            return coeff, concat_feat_mean, concat_feat_std, coeff

        else:
            if sp.issparse(self.adj):
                self.adj = sparse_mx_to_torch_sparse_tensor(self.adj)
            else:
                self.adj = torch.FloatTensor(self.adj)

            self.adj = self.adj.to(self.device)

            if is_sparse_tensor(self.adj):
                self.adj = normalize_adj_tensor(self.adj, sparse=True)
            else:
                self.adj = normalize_adj_tensor(self.adj)
            self.adj = SparseTensor(row=self.adj._indices()[0], col=self.adj._indices()[1],value=self.adj._values(), sparse_sizes=self.adj.size()).t()

            pseudo_y = torch.full_like(self.local_data.y, -1).to(self.device)
            pseudo_y[self.local_data.train_idx] = self.local_data.y[self.local_data.train_idx]
            aug_train_idx = copy.deepcopy(self.local_data.train_idx).to(self.device)

            if self.args.topk > 0:
                _, major_class_idx = torch.topk(self.glocal_class_coeff_tensor.squeeze(-1), largest=False, k=self.args.topk)
                pseudo_pred = self.lp_res / (torch.sum(self.lp_res, dim=1, keepdim=True) + 1e-9)
                pseudo_pred_val, pseudo_pred_ind = pseudo_pred.max(1)
                if self.args.d_thres > 0:
                    logging.info("with degree flitering")
                    local_node_degree = torch.bincount(self.local_data.edge_index[0], minlength=self.local_data.x.shape[0]).to(self.device)
                    mask = (pseudo_pred_val >= self.args.thres) & torch.isin(pseudo_pred_ind, major_class_idx) & (~aug_train_idx) & (local_node_degree >= self.args.d_thres)
                else:
                    mask = (pseudo_pred_val >= self.args.thres) & torch.isin(pseudo_pred_ind, major_class_idx) & (~aug_train_idx)
                logging.info(f"client: {self.client_id}, topk: {self.args.topk}, thres: {self.args.thres}, major class: {major_class_idx}, new labeled node: {torch.sum(mask).item()}")

                if mask.any():
                    aug_train_idx = aug_train_idx | mask
                    pseudo_y[mask] = pseudo_pred_ind[mask]

            self.propagate_model.eval()
            feat_list = self.local_data.x[aug_train_idx].to(self.device)
            tmp = self.local_data.x.to(self.device)
            for _ in range(self.args.hops):
                aggr_x = self.propagate_model.convs[0].propagate(self.adj.to(self.device), x=tmp).detach()
                feat_list = torch.cat((feat_list, aggr_x[aug_train_idx]), dim=1).to(self.device)
                tmp = aggr_x

            labels = pseudo_y[aug_train_idx]
            unique_labels, counts = torch.unique(labels, return_counts=True)
            unique_labels, counts = unique_labels.tolist(), counts.tolist()

            origin_labels = self.local_data.y[self.local_data.train_idx]
            origin_unique_labels, origin_counts = torch.unique(origin_labels, return_counts=True)
            origin_unique_labels, origin_counts = origin_unique_labels.tolist(), origin_counts.tolist()
            origin_coeff = {}
            for i, lbl in enumerate(origin_unique_labels):
                index = torch.where(origin_labels==lbl)
                if origin_counts[i] > 1:
                    origin_coeff[lbl] = origin_counts[i]

            coeff = {}
            concat_feat_mean = {}
            concat_feat_std = {}
            for i, lbl in enumerate(unique_labels):
                index = torch.where(labels==lbl)
                if counts[i] > 1:
                    coeff[lbl] = counts[i]
                    concat_feat_mean[lbl] = feat_list[index].mean(dim=0).to(self.device)
                    concat_feat_std[lbl] = feat_list[index].std(dim=0).to(self.device)

            return coeff, concat_feat_mean, concat_feat_std, origin_coeff
        


    def evaluate(self, model=None, debug=False):
        if model is None:
            self.model.eval()
            logits = self.model.forward(self.local_data.x, self.local_data.edge_index)
        else:
            model.eval()
            logits = model.forward(self.local_data.x, self.local_data.edge_index)

        loss_ce_fn = nn.CrossEntropyLoss()
        loss_train = loss_ce_fn(logits[self.local_data.train_idx], 
                        self.local_data.y[self.local_data.train_idx])
        loss_val = loss_ce_fn(logits[self.local_data.val_idx], 
                        self.local_data.y[self.local_data.val_idx])
        loss_test = loss_ce_fn(logits[self.local_data.test_idx], 
                        self.local_data.y[self.local_data.test_idx])
        acc_train_class, acc_train = accuracy(logits[self.local_data.train_idx], 
                        self.local_data.y[self.local_data.train_idx], by_class=True)
        acc_val_class, acc_val = accuracy(logits[self.local_data.val_idx], 
                        self.local_data.y[self.local_data.val_idx], by_class=True)
        acc_test_class, acc_test = accuracy(logits[self.local_data.test_idx], 
                        self.local_data.y[self.local_data.test_idx], by_class=True)
        f1_class_train, f1_macro_train = f1_score(logits[self.local_data.train_idx], 
                        self.local_data.y[self.local_data.train_idx], self.num_classes)
        f1_class_val, f1_macro_val = f1_score(logits[self.local_data.val_idx], 
                        self.local_data.y[self.local_data.val_idx], self.num_classes)
        f1_class_test, f1_macro_test = f1_score(logits[self.local_data.test_idx], 
                        self.local_data.y[self.local_data.test_idx], self.num_classes)
        # num_samples = torch.sum(self.local_data.test_idx)
        num_samples = self.local_data.x.shape[0]
        if debug:
            logging.info(f"[client {self.client_id}]: num nodes: {num_samples}, acc_train: {acc_train:.2f}, acc_val: {acc_val:.2f}, acc_test: {acc_test:.2f}, f1_macro_train: {f1_macro_train:.4f}, f1_macro_val: {f1_macro_val:.4f}, f1_macro_test: {f1_macro_test:.4f}, loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}, loss_test: {loss_test:.4f}")
            logging.info(f"f1 train: {f1_class_train}, f1 test: {f1_class_test}, acc test: {acc_test_class}")
        
        return num_samples, acc_val, acc_test, f1_macro_val, f1_macro_test

    def evaluate_inductive(self, model=None, debug=False):
        if model is None:
            self.model.eval()
            logits_train = self.model.forward(self.local_data.x, self.local_data.edge_index_train)[self.local_data.train_idx]
            logits_val = self.model.forward(self.local_data.x, self.local_data.edge_index_val)[self.local_data.val_idx]
            logits_test = self.model.forward(self.local_data.x, self.local_data.edge_index_test)[self.local_data.test_idx]
        else:
            model.eval()
            logits_train = model.forward(self.local_data.x, self.local_data.edge_index_train)[self.local_data.train_idx]
            logits_val = model.forward(self.local_data.x, self.local_data.edge_index_val)[self.local_data.val_idx]
            logits_test = model.forward(self.local_data.x, self.local_data.edge_index_test)[self.local_data.test_idx]

        loss_ce_fn = nn.CrossEntropyLoss()
        loss_train = loss_ce_fn(logits_train, 
                        self.local_data.y[self.local_data.train_idx])
        loss_val = loss_ce_fn(logits_val, 
                        self.local_data.y[self.local_data.val_idx])
        loss_test = loss_ce_fn(logits_test, 
                        self.local_data.y[self.local_data.test_idx])
        acc_train_class, acc_train = accuracy(logits_train, 
                        self.local_data.y[self.local_data.train_idx], by_class=True)
        acc_val_class, acc_val = accuracy(logits_val, 
                        self.local_data.y[self.local_data.val_idx], by_class=True)
        acc_test_class, acc_test = accuracy(logits_test, 
                        self.local_data.y[self.local_data.test_idx], by_class=True)
        f1_class_train, f1_macro_train = f1_score(logits_train, 
                        self.local_data.y[self.local_data.train_idx], self.num_classes)
        f1_class_val, f1_macro_val = f1_score(logits_val, 
                        self.local_data.y[self.local_data.val_idx], self.num_classes)
        f1_class_test, f1_macro_test = f1_score(logits_test, 
                        self.local_data.y[self.local_data.test_idx], self.num_classes)
        num_samples = self.local_data.x.shape[0]
        if debug:
            logging.info(f"[client {self.client_id}]: num nodes: {num_samples}, acc_train: {acc_train:.2f}, acc_val: {acc_val:.2f}, acc_test: {acc_test:.2f}, f1_macro_train: {f1_macro_train:.4f}, f1_macro_val: {f1_macro_val:.4f}, f1_macro_test: {f1_macro_test:.4f}, loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}, loss_test: {loss_test:.4f}")
            logging.info(f"f1 train: {f1_class_train}, f1 test: {f1_class_test}, acc test: {acc_test_class}")
        
        return num_samples, acc_val, acc_test, f1_macro_val, f1_macro_test
    
    def receive_model(self, model):
        self.model = copy.deepcopy(model).to(self.device)

    def receive_graph(self, feat_syn, edge_index_syn, edge_weight_syn, label_syn):
        self.pseudo_graph = {}
        self.pseudo_graph["feat"] = feat_syn
        self.pseudo_graph["edge_index"] = edge_index_syn
        self.pseudo_graph["edge_weight"] = edge_weight_syn
        self.pseudo_graph["label"] = label_syn

    def train_with_graph(self):
        if self.args.method == "hete":
            self.model = copy.deepcopy(self.init_model)
            self.model.reset_parameters()

        optimizer = Adam(self.model.parameters(), lr=self.args.lr_validation_model)
        best_val_acc = 0.
        best_test_acc = 0.
        best_f1_test = 0.
        best_model = None
        patience = 0
        loss_ce_fn = nn.CrossEntropyLoss()
        for j in range(self.args.validation_loop+1):
            self.model.train()
            optimizer.zero_grad()
            output_syn = self.model.forward(self.pseudo_graph["feat"], self.pseudo_graph["edge_index"], edge_weight=self.pseudo_graph["edge_weight"])
            loss = loss_ce_fn(output_syn, self.pseudo_graph["label"])
            loss.backward()
            optimizer.step()
            if j >= self.args.val_stage and j % self.args.val_stage == 0:
                _, acc_val, acc_test, f1_macro_val, f1_macro_test = self.evaluate()
                logging.info(f"Client {self.client_id}, train with pseudo graph loop: {j}, local acc val: {acc_val:.4f}, local acc test: {acc_test:.4f}, local f1_macro val: {f1_macro_val:.4f}, local f1_macro test: {f1_macro_test:.4f}")

                if acc_val > best_val_acc:
                    patience = 0
                    best_val_acc = acc_val
                    best_test_acc = acc_test
                    best_f1_test = f1_macro_test
                    best_model = copy.deepcopy(self.model)
                else:
                    patience += 1
                    if patience >= self.args.patience:
                        logging.info(f"Client {self.client_id}, train with pseudo graph ends at loop: {j}, best acc val: {best_val_acc:.4f}, best acc test: {best_test_acc:.4f}, best f1_macro test: {best_f1_test:.4f}")
                        break
        
        if best_model is not None:
            self.model = copy.deepcopy(best_model)

        return copy.deepcopy(self.model)
    
    def label_prop(self):
        model = MyLabelProp(num_layers=self.args.lp_iter, alpha=self.args.lp_alpha).to(self.device)
        out = model(self.local_data.y, self.local_data.edge_index, mask=self.local_data.train_idx, num_classes=self.num_classes)

        mask = (out.sum(dim=1) == 0)
        num_classes = out.size(1)
        uniform_distribution = torch.full((mask.sum().item(), num_classes), 1.0 / num_classes).to(self.device)
        out[mask] = uniform_distribution

        return out
    
    def label_homophily_details(self):
        num_nodes = self.local_data.num_nodes
        homophily_class = {c: 0. for c in range(self.num_classes)}
        homophily_node = []
        num_per_class = {c: 0 for c in range(self.num_classes)}
        
        for nd_u in range(num_nodes):
            if self.local_data.train_idx[nd_u] == 1:
                homophily = 0.
                hit = 0.
                tot = 0.
                nd_v_list = self.local_data.edge_index[1, :][self.local_data.edge_index[0, :] == nd_u]
                if len(nd_v_list) != 0:
                    for i in range(len(nd_v_list)):
                        nd_v = nd_v_list[i]
                        if self.local_data.train_idx[nd_v] != 1:
                            continue 

                        if self.local_data.y[nd_u] == self.local_data.y[nd_v]:
                            hit += 1.
                        tot += 1.
                    if tot > 0:
                        homophily = hit / tot

                homophily_node.append(homophily)
                homophily_class[self.local_data.y[nd_u].item()] += homophily
                num_per_class[self.local_data.y[nd_u].item()] += 1
            else:
                homophily_node.append(1)

        return homophily_node, homophily_class, num_per_class

    def train_ft_reg_nd(self):
        if self.args.method == "hete":
            self.model = copy.deepcopy(self.init_model)
            self.model.reset_parameters()

        optimizer = Adam(self.model.parameters(), lr=self.args.lr_validation_model)
        best_val_acc = 0.
        best_test_acc = 0.
        best_f1_test = 0.
        best_model = None
        patience = 0

        loss_ce_fn = nn.CrossEntropyLoss()
        for j in range(self.args.validation_loop+1):
            self.model.train()
            optimizer.zero_grad()
            output_syn = self.model.forward(self.pseudo_graph["feat"], self.pseudo_graph["edge_index"], edge_weight=self.pseudo_graph["edge_weight"])
            loss = loss_ce_fn(output_syn, self.pseudo_graph["label"])
            loss.backward()
            optimizer.step()
            if j >= self.args.val_stage and j % self.args.val_stage == 0:
                if self.args.dataset in ["Reddit", "Flickr", "Reddit2"]:
                    _, acc_val, acc_test, f1_macro_val, f1_macro_test = self.evaluate_inductive(debug= j == self.args.validation_loop)
                else:
                    _, acc_val, acc_test, f1_macro_val, f1_macro_test = self.evaluate(debug= j == self.args.validation_loop)
                logging.info(f"Client {self.client_id}, train with pseudo graph loop: {j}, local acc val: {acc_val:.4f}, local acc test: {acc_test:.4f}, local f1_macro val: {f1_macro_val:.4f}, local f1_macro test: {f1_macro_test:.4f}")

                if acc_val > best_val_acc:
                    patience = 0
                    best_val_acc = acc_val
                    best_test_acc = acc_test
                    best_f1_test = f1_macro_test
                    best_model = copy.deepcopy(self.model)
                else:
                    patience += 1
                    if patience >= self.args.patience:
                        logging.info(f"Client {self.client_id}, train with pseudo graph loop: {j}, best acc val: {best_val_acc:.4f}, best acc test: {best_test_acc:.4f}, best f1_macro test: {best_f1_test:.4f}\n")
                        break


        if best_model is not None:
            self.model = copy.deepcopy(best_model)
            if self.args.dataset in ["Reddit", "Flickr", "Reddit2"]:
                _ = self.evaluate_inductive(debug=True)
            else:
                _ = self.evaluate(debug=True)

        local_model = copy.deepcopy(self.model)
        local_model.eval()
        for param in list(local_model.parameters()):
            param.requires_grad = False


        global_logits = local_model.forward(self.local_data.x, self.local_data.edge_index).detach()

        kl_loss = nn.KLDivLoss(reduction="none")
        ft_optimizer = Adam(self.model.parameters(), lr=self.args.lr_ft)
        best_val_acc = 0.
        best_test_acc = 0.
        best_f1_test = 0.
        best_model = None
        patience = 0

        for j in range(self.args.validation_loop+1):
            self.model.train()
            ft_optimizer.zero_grad()
            logits = self.model.forward(self.local_data.x, self.local_data.edge_index)
            loss1 = loss_ce_fn(logits[self.local_data.train_idx], self.local_data.y[self.local_data.train_idx])
            loss2 = self.local_nd_coeff * kl_loss(F.log_softmax(logits, dim=1), F.softmax(global_logits, dim=1))
            loss2 = loss2.sum() / logits.size(0)
            loss = loss1 + loss2
            loss.backward()
            ft_optimizer.step()
            if j >= self.args.val_stage and j % self.args.val_stage == 0:
                logging.info(f"Client {self.client_id}, finetuning with reg loop: {j}, loss1: {loss1:.6f}, loss2: {loss2:.6f}, loss: {loss:.6f}")
                if self.args.dataset in ["Reddit", "Flickr", "Reddit2"]:
                    _, acc_val, acc_test, f1_macro_val, f1_macro_test = self.evaluate_inductive()
                else:
                    _, acc_val, acc_test, f1_macro_val, f1_macro_test = self.evaluate()
                logging.info(f"Client {self.client_id}, finetuning with reg loop: {j}, local acc val: {acc_val:.4f}, local acc test: {acc_test:.4f}, local f1_macro val: {f1_macro_val:.4f}, local f1_macro test: {f1_macro_test:.4f}")
                if acc_val > best_val_acc:
                    patience = 0
                    best_val_acc = acc_val
                    best_test_acc = acc_test
                    best_f1_test = f1_macro_test
                    best_model = copy.deepcopy(self.model)
                else:
                    patience += 1
                    if patience >= self.args.patience:
                        logging.info(f"Client {self.client_id}, finetune ends at loop: {j}, best acc val: {best_val_acc:.4f}, best acc test: {best_test_acc:.4f}, best f1_macro test: {best_f1_test:.4f}\n")
                        break

        if best_model is not None:
            self.model = copy.deepcopy(best_model)
        return copy.deepcopy(self.model)
    

class Server:
    def __init__(self, args, model, dataset, clients_list, client_model_name_list, propagate_model, device):
        self.args = args
        self.model = copy.deepcopy(model)
        if self.model is not None:
            self.model = self.model.to(device)
        self.dataset = dataset
        self.clients = clients_list
        self.num_classes = self.dataset.num_classes
        self.client_model_name_list = client_model_name_list
        self.feat_dim = self.dataset.global_dataset.x.shape[1]
        self.propagate_model = propagate_model
        self.device = device

    def gather_dist(self):

        server_mean = {lbl: 0 for lbl in range(self.num_classes)}
        server_mean_list = {lbl: [] for lbl in range(self.num_classes)}
        server_std = {lbl: 0 for lbl in range(self.num_classes)}
        server_std_list = {lbl: [] for lbl in range(self.num_classes)}
        server_coeff = {lbl: 0 for lbl in range(self.num_classes)}
        server_coeff_list = {lbl: [] for lbl in range(self.num_classes)}
        server_origin_coeff = {lbl: 0 for lbl in range(self.num_classes)}

        for client in self.clients:
            coeff, concat_feat_mean, concat_feat_std, origin_coeff = client.get_distribution()
            logging.info(f"client {client.client_id}'s label set: {coeff}, original label set: {origin_coeff}")
            for lbl in coeff.keys():
                server_coeff[lbl] += coeff[lbl]
                server_coeff_list[lbl].append(coeff[lbl])
                server_origin_coeff[lbl] += origin_coeff[lbl]
                server_mean_list[lbl].append(concat_feat_mean[lbl])
                server_mean[lbl] += concat_feat_mean[lbl] * coeff[lbl]
                server_std_list[lbl].append(concat_feat_std[lbl])

        logging.info(server_coeff)
        logging.info(server_origin_coeff)
        for lbl in server_coeff.keys():
            if server_coeff[lbl] == 0:
                continue
            server_mean[lbl] /= server_coeff[lbl]
            weighted_variance_sum = sum((n - 1) * s**2 + n * ((m - server_mean[lbl])**2) for n, s, m in zip(server_coeff_list[lbl], server_std_list[lbl], server_mean_list[lbl]))
            original_variance = weighted_variance_sum / (server_coeff[lbl] - len(server_coeff_list[lbl]))
            server_std[lbl] = torch.sqrt(original_variance)
        
        sum_coeff = sum(server_coeff.values())
        sum_origin_coeff = sum(server_origin_coeff.values())
        logging.info(f"sum coeff: {sum_coeff}, sum origin coeff: {sum_origin_coeff}")
        ipc_dict = {lbl: 0 for lbl in range(self.num_classes)}
        labels_syn = []
        for lbl in range(self.num_classes):
            if self.args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
                ipc_dict[lbl] = self.args.ipc
            else:
                ipc_dict[lbl] = math.ceil(server_origin_coeff[lbl] * self.args.rate)
            
            labels_syn += [lbl] * ipc_dict[lbl]

        logging.info(f"##### Will condense {ipc_dict} for each class")
        logging.info(f"synthesized label set: {labels_syn}, tot num = {len(labels_syn)}")
        labels_syn = torch.LongTensor(labels_syn).to(self.device)
        nnodes_syn = len(labels_syn)
        feat_syn = nn.Parameter(torch.FloatTensor(nnodes_syn, self.feat_dim).to(device))
        feat_syn.data.copy_(torch.randn(feat_syn.size()))
        # with torch.no_grad():
        #     feat_syn.data.copy_(torch.randn(feat_syn.size()).to(self.device))
        pge = PGE(nfeat=self.feat_dim, nnodes=nnodes_syn, device=self.device, args=self.args).to(device)

        optimizer_feat = Adam([feat_syn], lr=self.args.lr_feat)
        optimizer_pge = Adam(pge.parameters(), lr=self.args.lr_adj)
        overall_best_val = 0.
        overall_best_test = 0.
        overall_best_f1_test = 0.
        overall_best_f1_val = 0.
        for i in range(self.args.condensing_loop+1):
            optimizer_pge.zero_grad()
            optimizer_feat.zero_grad()
            
            adj_syn = pge(feat_syn).to(device)
            adj_syn[adj_syn<args.threshold] = 0
            edge_index_syn = torch.nonzero(adj_syn).T
            edge_weight_syn = adj_syn[edge_index_syn[0], edge_index_syn[1]]

            # smoothness loss
            feat_difference = torch.exp(-0.5 * torch.pow(feat_syn[edge_index_syn[0]] - feat_syn[edge_index_syn[1]], 2))
            smoothness_loss = torch.dot(edge_weight_syn, torch.mean(feat_difference,1).flatten()) / torch.sum(edge_weight_syn)

            self.propagate_model.eval()
            edge_index_syn, edge_weight_syn = gcn_norm(edge_index_syn, edge_weight_syn, nnodes_syn)
            feat_list = feat_syn.to(self.device)
            tmp = feat_list
            for _ in range(self.args.hops):
                aggr_x = self.propagate_model.convs[0].propagate(edge_index_syn, x=tmp, edge_weight=edge_weight_syn)
                feat_list = torch.cat((feat_list, aggr_x), dim=1).to(self.device)
                tmp = aggr_x

            # alignment loss
            alignment_loss = torch.tensor(0.0).to(self.device)
            loss_mse_fn = nn.MSELoss()
            for c in range(self.num_classes):
                if server_coeff[c] == 0:
                    continue
                index=torch.where(labels_syn == c)
                feat_mean_loss = (server_coeff[c] / sum_coeff) * loss_mse_fn(server_mean[c], feat_list[index].mean(dim=0))
                feat_std_loss = (server_coeff[c] / sum_coeff) * loss_mse_fn(server_std[c], feat_list[index].std(dim=0))
                if feat_syn[index].shape[0] != 1:
                    alignment_loss += (feat_mean_loss + feat_std_loss)
                else:
                    alignment_loss += (feat_mean_loss)

            # total loss
            loss = self.args.feat_alpha * alignment_loss + self.args.smoothness_alpha * smoothness_loss
            loss.backward()
            if i % 50 < 10:
                optimizer_pge.step()
            else:
                optimizer_feat.step()

            if i >= 200 and i % 200 == 0:
                logging.info(f"loop {i}, condensation loss: {loss.item():.6f}, alignment loss: {alignment_loss:.6f}, smoothness loss: {smoothness_loss:.6f}")
            
        
        adj_syn = pge.inference(feat_syn).detach().to(self.device)
        adj_syn[adj_syn<self.args.threshold] = 0
        adj_syn.requires_grad = False
        edge_index_syn = torch.nonzero(adj_syn).T
        edge_weight_syn = adj_syn[edge_index_syn[0], edge_index_syn[1]]
        if self.args.method == 'homo':
            self.model.reset_parameters()
        for client in self.clients:
            if self.args.method == 'homo':
                client.receive_model(copy.deepcopy(self.model))
            client.receive_graph(copy.deepcopy(feat_syn), copy.deepcopy(edge_index_syn), copy.deepcopy(edge_weight_syn), copy.deepcopy(labels_syn))

        local_models = []
        for client in self.clients:
            if self.args.local_strategy == 'global':
                local_models.append(client.train_with_graph())
            elif 'nd_lp_reg_ft' in self.args.local_strategy:
                local_models.append(client.train_ft_reg_nd())

            
        if self.args.personalized:
            logging.info(f"evaluate personality:")
            acc_val_avg, acc_test_avg, f1_macro_val_avg, f1_macro_test_avg, acc_test_list, f1_macro_test_list = self.evaluate_global_to_local()
            logging.info(f"acc val avg: {acc_val_avg:.4f}, acc test avg: {acc_test_avg:.4f}, f1_macro val avg: {f1_macro_val_avg:.4f}, f1_macro test avg: {f1_macro_test_avg:.4f}")
            logging.info(f"acc test list: {acc_test_list}, f1_macro test list: {f1_macro_test_list}")            
        else:
            logging.info(f"evaluate generality:")
            acc_val_list = []
            acc_test_list = []
            f1_test_list = []
            f1_val_list = []
            for client in self.clients:
                acc_val, acc_test, f1_macro_val, f1_macro_test, _, _ = self.evaluate_global_to_local(local_models[client.client_id])
                acc_val_list.append(acc_val)
                acc_test_list.append(acc_test)
                f1_test_list.append(f1_macro_test)
                f1_val_list.append(f1_macro_val)

            acc_val_avg = sum(acc_val_list) / len(acc_val_list)
            acc_test_avg = sum(acc_test_list) / len(acc_test_list)
            f1_macro_test_avg = sum(f1_test_list) / len(f1_test_list)
            f1_macro_val_avg = sum(f1_val_list) / len(f1_val_list)
            logging.info(f"acc val avg: {acc_val_avg:.4f}, acc test avg: {acc_test_avg:.4f}, f1_macro val avg: {f1_macro_val_avg:.4f}, f1_macro test avg: {f1_macro_test_avg:.4f}, acc val list: {acc_val_list}, acc test list: {acc_test_list}")            
        
        if acc_val_avg > overall_best_val:
            overall_best_val = acc_val_avg
            overall_best_test = acc_test_avg
        if f1_macro_val_avg > overall_best_f1_val:
            overall_best_f1_val = f1_macro_val_avg
            overall_best_f1_test = f1_macro_test_avg

        logging.info(f"best val acc: {overall_best_val:.4f}, best test acc: {overall_best_test:.4f}, best val f1 macro: {overall_best_f1_val:.4f}, best test f1 macro: {overall_best_f1_test:.4f}")
        acc_1_list.append(overall_best_test)
        f1_1_list.append(overall_best_f1_test)

    def select_clients(self):
        return (
            self.clients if self.args.frac == 1.0
            else random.sample(self.clients, int(round(len(self.clients) * self.args.frac)))
        )

    def evaluate_global_to_local(self, model=None):
        tot_samples = 0
        tot_acc_val = 0.
        tot_acc_test = 0.
        tot_f1_macro_test = 0.
        tot_f1_macro_val = 0.
        acc_test_list = []
        f1_macro_test_list = []
        f1_macro_val_list = []
        for client in self.clients:
            if model is None:
                if self.args.dataset in ["Reddit", "Flickr", "Reddit2"]:
                    num_samples, acc_val, acc_test, f1_macro_val, f1_macro_test = client.evaluate_inductive(debug=True)
                else:
                    num_samples, acc_val, acc_test, f1_macro_val, f1_macro_test = client.evaluate(debug=True)
            else:
                if self.args.dataset in ["Reddit", "Flickr", "Reddit2"]:
                    num_samples, acc_val, acc_test, f1_macro_val, f1_macro_test = client.evaluate_inductive(model, debug=True)
                else:
                    num_samples, acc_val, acc_test, f1_macro_val, f1_macro_test = client.evaluate(model, debug=True)

            acc_test_list.append(acc_test)
            f1_macro_test_list.append(f1_macro_test)
            f1_macro_val_list.append(f1_macro_val)
            tot_samples += num_samples
            tot_acc_val += acc_val * num_samples
            tot_acc_test += acc_test * num_samples
            tot_f1_macro_test += f1_macro_test * num_samples
            tot_f1_macro_val += f1_macro_val * num_samples

        return tot_acc_val / tot_samples, tot_acc_test / tot_samples, tot_f1_macro_val / tot_samples, tot_f1_macro_test / tot_samples, acc_test_list, f1_macro_test_list
    
    def evaluate_global_to_global(self):
        pass


acc_1_list = []
f1_1_list = []


if __name__ == "__main__":

    os.environ['NUMEXPR_MAX_THREADS'] = r'8'
    
    if args.method == 'homo':
        use_model_name = args.validation_model
        model_name = args.validation_model
        if args.validation_model in ['GCN', 'MLP', 'GIN', 'Sage']:
            use_model_name += f'_L{args.nlayers}_hid{args.hid_dim}'
            model_name += f'_{args.nlayers}_{args.hid_dim}'
        elif args.validation_model in ['SGC',]:
            use_model_name += f'_K{args.K}'
            model_name += f'_{args.K}'
        elif args.validation_model in ['SIGN',]:
            use_model_name += f'_K{args.K}_L{args.nlayers}_hid{args.hid_dim}'
            model_name += f'_{args.K}_{args.nlayers}_{args.hid_dim}'

        client_model_name_list = [model_name for _ in range(args.num_clients)]

        log_format = '%(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
        if args.personalized:
            if 'lp' in args.local_strategy:
                args.local_strategy += f"_{args.lp_iter}_{args.lp_alpha}"
            if 'reg' in args.local_strategy:
                args.local_strategy += f"_{args.reg_a}"
                if args.reg_max is not None:
                    args.local_strategy += f"_u{args.reg_max}" # upper
                if args.reg_min is not None:
                    args.local_strategy += f"_l{args.reg_min}" # lower
            if args.dataset in ["Cora", "CiteSeer", "PubMed", ]:
                config_string = f"p{args.local_strategy}_{args.method}_{args.dataset}_{args.partition}_{args.num_clients}_ipc{args.ipc}_h{args.hops}_{use_model_name}_"
            elif args.dataset in ["ogbn-arxiv", "ogbn-products", "Flickr", "Reddit", "Reddit2"]:
                config_string = f"p{args.local_strategy}_{args.method}_{args.dataset}_{args.partition}_{args.num_clients}_r{args.rate}_h{args.hops}_{use_model_name}_"
        else:
            if args.dataset in ["Cora", "CiteSeer", "PubMed", ]:
                config_string = f"{args.method}_{args.dataset}_{args.partition}_{args.num_clients}_ipc{args.ipc}_h{args.hops}_{use_model_name}_"
            elif args.dataset in ["ogbn-arxiv", "ogbn-products", "Flickr", "Reddit", "Reddit2"]:
                config_string = f"{args.method}_{args.dataset}_{args.partition}_{args.num_clients}_r{args.rate}_h{args.hops}_{use_model_name}_"
        log_file = f'main_{args.condensing_loop}_{args.topk}_{args.thres}_{args.d_thres}' + config_string + f'{args.lr_validation_model}_{args.lr_ft}_' + 'log.txt'
        log_path = os.path.join(args.save_root_path, log_file)
        print(log_path)
        if os.path.exists(log_path):
            raise Exception('log file already exists!')
        fh = logging.FileHandler(log_path, mode='w')
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

        for runs in range(3):
            seed_everything(seed=runs)
            logging.info(f"\nlog path: {log_path}, Run {runs} begins...")
            dataset = load_dataset(args)
            device = torch.device(f"cuda:{args.gpu_id}")
            subgraphs = [dataset.subgraphs[client_id].to(device) for client_id in range(args.num_clients)]

            if args.validation_model == "GCN":
                # global_model = GCN(in_channels=subgraphs[0].x.shape[1], hidden_channels=args.hid_dim, num_layers=args.nlayers, out_channels=dataset.num_classes, dropout=0).to(device)
                global_model = MyGCN(in_channels=subgraphs[0].x.shape[1], hidden_channels=args.hid_dim, num_layers=args.nlayers, out_channels=dataset.num_classes, dropout=0).to(device)
            elif args.validation_model == 'SGC':
                global_model = SGConv(in_channels=subgraphs[0].x.shape[1], out_channels=dataset.num_classes, K=args.K).to(device)
            elif args.validation_model == "SIGN":
                global_model = SIGN(in_channels=subgraphs[0].x.shape[1], hidden_channels=args.hid_dim, num_layers=args.nlayers, out_channels=dataset.num_classes, K=args.K, dropout=0).to(device)
            propagate_model = GCN(in_channels=subgraphs[0].x.shape[1], hidden_channels=args.hid_dim, num_layers=1, out_channels=dataset.num_classes, dropout=0).to(device)
            clients_list = [Client(args, i, copy.deepcopy(global_model), client_model_name_list[i], subgraphs[i], copy.deepcopy(propagate_model), dataset.num_classes, device) for i in range(args.num_clients)]

            server = Server(args, copy.deepcopy(global_model), dataset, clients_list, client_model_name_list, copy.deepcopy(propagate_model), device)

            print("Create Clients and Server finished")

            logging.info(global_model)
            logging.info(f"{args.__dict__}")

            server.gather_dist()

        logging.info(f"{acc_1_list}")
        logging.info(f"{f1_1_list}")
        acc_1_np = np.array(acc_1_list)
        f1_1_np = np.array(f1_1_list)
        logging.info(f"acc@1: {acc_1_np.mean()}, {acc_1_np.std()}")
        logging.info(f"f1@1: {f1_1_np.mean()}, {f1_1_np.std()}")

    elif args.method == 'hete':
        client_model_name_list = ["SSGC_2_2_64", "GCN_2_64", "GCN_2_128", "GCN_2_256", "GCN_2_512", "SGC_2", "GCN_3_64", "GCN_3_512", "APPNP_2_2_64", "SGC_4"]
        use_model_name = "-".join(client_model_name_list)
        log_format = '%(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
        if args.personalized:
            if 'lp' in args.local_strategy:
                args.local_strategy += f"_{args.lp_iter}_{args.lp_alpha}"
            if 'reg' in args.local_strategy:
                args.local_strategy += f"_{args.reg_a}"
                if args.reg_max is not None:
                    args.local_strategy += f"_u{args.reg_max}" # upper
                if args.reg_min is not None:
                    args.local_strategy += f"_l{args.reg_min}" # lower
            if args.dataset in ["Cora", "CiteSeer", "PubMed", ]:
                config_string = f"p{args.local_strategy}_{args.method}_{args.dataset}_{args.partition}_{args.num_clients}_ipc{args.ipc}_h{args.hops}_{use_model_name}_"
            elif args.dataset in ["ogbn-arxiv", "ogbn-products", "Flickr", "Reddit", "Reddit2"]:
                config_string = f"p{args.local_strategy}_{args.method}_{args.dataset}_{args.partition}_{args.num_clients}_r{args.rate}_h{args.hops}_{use_model_name}_"
        else:
            if args.dataset in ["Cora", "CiteSeer", "PubMed", ]:
                config_string = f"{args.method}_{args.dataset}_{args.partition}_{args.num_clients}_ipc{args.ipc}_h{args.hops}_{use_model_name}_"
            elif args.dataset in ["ogbn-arxiv", "ogbn-products", "Flickr", "Reddit", "Reddit2"]:
                config_string = f"{args.method}_{args.dataset}_{args.partition}_{args.num_clients}_r{args.rate}_h{args.hops}_{use_model_name}_"
        if args.d_thres > 0:
            log_file = f'main_{args.condensing_loop}_{args.topk}_{args.thres}_{args.d_thres}' + config_string + f'{args.lr_validation_model}_{args.lr_ft}_' + 'log.txt'
        else:
            log_file = f'main_{args.condensing_loop}_{args.topk}_{args.thres}_' + config_string + f'{args.lr_validation_model}_{args.lr_ft}_' + 'log.txt'
        log_path = os.path.join(args.save_root_path, log_file)
        print(log_path)
        if os.path.exists(log_path):
            raise Exception('log file already exists!')
        fh = logging.FileHandler(log_path, mode='w')
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

        for runs in range(3):
            logging.info(f"\nRun {runs} begins...")
            seed_everything(seed=runs)
            dataset = load_dataset(args)
            device = torch.device(f"cuda:{args.gpu_id}")
            subgraphs = [dataset.subgraphs[client_id].to(device) for client_id in range(args.num_clients)]
            local_model_list = []
            for model_name in client_model_name_list:
                if "GCN" in model_name:
                    model_name_config = model_name.split('_')
                    nlayers = int(model_name_config[1])
                    hid_dim = int(model_name_config[2])
                    local_model_list.append(GCN(in_channels=subgraphs[0].x.shape[1], hidden_channels=hid_dim, num_layers=nlayers, out_channels=dataset.num_classes, dropout=0).to(device))
                elif 'SGC' in model_name:
                    model_name_config = model_name.split('_')
                    K = int(model_name_config[1])
                    local_model_list.append(SGConv(in_channels=subgraphs[0].x.shape[1], out_channels=dataset.num_classes, K=K).to(device))
                elif "SSGC" in model_name:
                    model_name_config = model_name.split('_')
                    K = int(model_name_config[1])
                    nlayers = int(model_name_config[2])
                    hid_dim = int(model_name_config[3])
                    local_model_list.append(SSGC(in_channels=subgraphs[0].x.shape[1], hidden_channels=hid_dim, num_layers=nlayers, out_channels=dataset.num_classes, K=K, dropout=0).to(device))
                elif "APPNP" in model_name:
                    model_name_config = model_name.split('_')
                    K = int(model_name_config[1])
                    nlayers = int(model_name_config[2])
                    hid_dim = int(model_name_config[3])
                    local_model_list.append(MyAPPNP(in_channels=subgraphs[0].x.shape[1], hidden_channels=hid_dim, num_layers=nlayers, out_channels=dataset.num_classes, K=K, dropout=0).to(device))
            
            propagate_model = GCN(in_channels=subgraphs[0].x.shape[1], hidden_channels=args.hid_dim, num_layers=1, out_channels=dataset.num_classes, dropout=0).to(device)
            clients_list = [Client(args, i, copy.deepcopy(local_model_list[i]), client_model_name_list[i], subgraphs[i], copy.deepcopy(propagate_model), dataset.num_classes, device) for i in range(args.num_clients)]

            server = Server(args, None, dataset, clients_list, client_model_name_list, copy.deepcopy(propagate_model), device)

            print("Create Clients and Server finished")

            logging.info(local_model_list)
            logging.info(f"{args.__dict__}")

            server.gather_dist()

        logging.info(f"{acc_1_list}")
        logging.info(f"{f1_1_list}")
        acc_1_np = np.array(acc_1_list)
        f1_1_np = np.array(f1_1_list)
        logging.info(f"acc@1: {acc_1_np.mean()}, {acc_1_np.std()}")
        logging.info(f"f1@1: {f1_1_np.mean()}, {f1_1_np.std()}")