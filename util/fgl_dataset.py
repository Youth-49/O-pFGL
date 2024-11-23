import os
import os.path as osp
import re
import time
import warnings
import torch
from torch_geometric.data import Dataset
from util.base_data_util import data_partition

warnings.filterwarnings('ignore')

class FGLDataset(Dataset):
    def __init__(
        self,
        args,
        root,
        name,
        num_clients,
        partition,
        train,
        val,
        test,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        part_delta=20,
        louvain_num_coms=10,
        metis_num_coms=10
    ):
        start = time.time()
        self.args = args
        self.root = root
        self.name = name
        self.num_clients = num_clients
        self.partition = partition
        self.train = train
        self.val = val
        self.test = test
        self.part_delta = part_delta
        self.louvain_num_coms = louvain_num_coms
        self.metis_num_coms = metis_num_coms

        super(FGLDataset, self).__init__(
            root, transform, pre_transform, pre_filter)
        self.load_data()

        end = time.time()
        print(f"load FGL dataset {name} done ({end-start:.2f} sec)")

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        fmt_name = re.sub("-", "_", self.name)
        return osp.join(
            self.raw_dir, fmt_name, "Client{}".format(
                self.num_clients), self.partition
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self) -> str:
        files_names = ["data{}.pt".format(i) for i in range(self.num_clients)]
        return files_names

    def download(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            osp.join(self.processed_dir, "data{}.pt".format(idx)))
        return data

    def process(self):
        self.load_global_graph()

        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        subgraph_list = data_partition(
            G=self.global_data,
            num_clients=self.num_clients,
            train=self.train,
            val=self.val,
            test=self.test,
            partition=self.partition,
            part_delta=self.part_delta,
            louvain_num_coms=self.louvain_num_coms,
            metis_num_coms=self.metis_num_coms,
        )

        for i in range(self.num_clients):
            torch.save(subgraph_list[i], self.processed_paths[i])

    def load_global_graph(self):
        if self.name in ["Cora", "CiteSeer", "PubMed"]:
            from torch_geometric.datasets import Planetoid
            self.global_dataset = Planetoid(root=self.raw_dir, name=self.name)
        elif self.name in ["ogbn-arxiv", "ogbn-products"]:
            from ogb.nodeproppred import PygNodePropPredDataset
            self.global_dataset = PygNodePropPredDataset(
                root=self.raw_dir, name=self.name
            )
        elif self.name in ["Reddit"]:
            from torch_geometric.datasets import Reddit
            self.global_dataset = Reddit(
                root=os.path.join(self.raw_dir, self.name))
        elif self.name in ["Reddit2"]:
            from torch_geometric.datasets import Reddit2
            self.global_dataset = Reddit2(
                root=os.path.join(self.raw_dir, self.name))
        elif self.name in ["Flickr"]:
            from torch_geometric.datasets import Flickr
            self.global_dataset = Flickr(
                root=os.path.join(self.raw_dir, self.name))
        else:
            raise ValueError(
                "Not supported for this dataset, please check root file path and dataset name"
            )
        self.global_data = self.global_dataset.data
        self.global_data.num_classes = self.global_dataset.num_classes


    def load_data(self):
        print("loading graph...")
        self.load_global_graph()
        self.feat_dim = self.global_dataset.num_features
        self.out_dim = self.global_dataset.num_classes
        self.global_data = self.global_dataset.data
        self.subgraphs = [self.get(i) for i in range(self.num_clients)]   
        for i in range(len(self.subgraphs)):
            self.subgraphs[i].feat_dim = self.global_dataset.num_features
            self.subgraphs[i].out_dim = self.out_dim
        if self.name in ["ogbn-arxiv", "ogbn-products"]:
            for i in range(self.num_clients):
                self.subgraphs[i].y = self.subgraphs[i].y.squeeze()

        def filter_edges(edge_index, node_idx):
            src, dst = edge_index[0], edge_index[1]
            mask = node_idx[src] & node_idx[dst]
            return edge_index[:, mask]
        
        if self.name in ["Reddit", "Flickr", "Reddit2"]:
            self.global_data.edge_index_train = filter_edges(self.global_data.edge_index, self.global_data.train_mask)
            self.global_data.edge_index_val = filter_edges(self.global_data.edge_index, self.global_data.val_mask)
            self.global_data.edge_index_test = filter_edges(self.global_data.edge_index, self.global_data.test_mask)
 
            for i in range(len(self.subgraphs)):
                self.subgraphs[i].edge_index_train = filter_edges(self.subgraphs[i].edge_index, self.subgraphs[i].train_idx)
                self.subgraphs[i].edge_index_val = filter_edges(self.subgraphs[i].edge_index, self.subgraphs[i].val_idx)
                self.subgraphs[i].edge_index_test = filter_edges(self.subgraphs[i].edge_index, self.subgraphs[i].test_idx)

