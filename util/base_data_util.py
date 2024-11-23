import time
import numpy as np
import scipy.sparse as sp
import random
import torch
from torch import Tensor
from torch_geometric.utils.convert import to_networkx
from louvain.community import community_louvain
from torch_geometric.data import Data
from sklearn.cluster import KMeans
import copy
import logging

def remove_self_loops(edge_index, edge_attr=None):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


def to_undirected(edge_index):
    if isinstance(edge_index, sp.csr_matrix) or isinstance(edge_index, sp.coo_matrix):
        row, col = edge_index.row, edge_index.col
        row, col = torch.from_numpy(row), torch.from_numpy(col)
    else:
        row, col = edge_index
        if not isinstance(row, Tensor) or not isinstance(col, Tensor):
            row, col = torch.from_numpy(row), torch.from_numpy(col)
    new_row = torch.hstack((row, col))
    new_col = torch.hstack((col, row))
    new_edge_index = torch.stack((new_row, new_col), dim=0)
    return new_edge_index


def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index.cpu().numpy()]


def idx_to_mask(index, size):
    mask = torch.zeros((size,), dtype=torch.bool)
    mask[index] = 1
    return mask


def louvain_partition(graph, num_clients, delta=20):
    num_nodes = graph.number_of_nodes()

    partition = community_louvain.best_partition(graph)

    groups = []

    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])
    print(groups)
    partition_groups = {group_i: [] for group_i in groups}

    for key in partition.keys():
        partition_groups[partition[key]].append(key)

    group_len_max = num_nodes // num_clients - delta
    for group_i in groups:
        while len(partition_groups[group_i]) > group_len_max:
            long_group = list.copy(partition_groups[group_i])
            partition_groups[group_i] = list.copy(long_group[:group_len_max])
            new_grp_i = max(groups) + 1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]
    print(groups)

    len_list = []
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))

    len_dict = {}

    for i in range(len(groups)):
        len_dict[groups[i]] = len_list[i]
    sort_len_dict = {
        k: v
        for k, v in sorted(len_dict.items(), key=lambda item: item[1], reverse=True)
    }

    owner_node_ids = {owner_id: [] for owner_id in range(num_clients)}

    owner_nodes_len = num_nodes // num_clients
    owner_list = [i for i in range(num_clients)]
    owner_ind = 0

    bad_key = 1000

    for group_i in sort_len_dict.keys():
        while (
            len(owner_list) >= 2
            and len(owner_node_ids[owner_list[owner_ind]]) >= owner_nodes_len
        ):
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        cnt = 0
        while (
            len(owner_node_ids[owner_list[owner_ind]]) +
                len(partition_groups[group_i])
            >= owner_nodes_len + delta
        ):
            owner_ind = (owner_ind + 1) % len(owner_list)
            cnt += 1
            if cnt > bad_key:
                cnt = 0
                min_v = 1e15
                for i in range(len(owner_list)):
                    if len(owner_node_ids[owner_list[owner_ind]]) < min_v:
                        min_v = len(owner_node_ids[owner_list[owner_ind]])
                        owner_ind = i
                break

        owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]
    node_dict = owner_node_ids

    print("end louvain")
    return node_dict


def data_partition(G, num_clients, train, val, test, partition, part_delta, louvain_num_coms, metis_num_coms):
    time_st = time.time()
    logging.info("start g to nxg")
    graph_nx = to_networkx(G, to_undirected=True, remove_self_loops=True)
    logging.info(f"get nxg {time.time()-time_st} sec.")

    if partition == "Louvain+":
        logging.info("Conducting louvain+ graph partition...")
        com_node_dict = louvain_partition(
            graph=graph_nx, num_clients=louvain_num_coms, delta=part_delta
        )
        communities = {}
        for com_id, node_list in com_node_dict.items():
            if len(node_list) == 0:
                continue
            if com_id not in communities:
                communities[com_id] = {"nodes":node_list, "num_nodes":len(node_list), "label_distribution":[0] * G.num_classes}
            
        for com_id in communities.keys():
            for node in communities[com_id]["nodes"]:
                label = copy.deepcopy(G.y[node])
                communities[com_id]["label_distribution"][label] += 1

        num_communities = len(communities)
        clustering_data = np.zeros(shape=(num_communities, G.num_classes))
        for com_id in communities.keys():
            for class_i in range(G.num_classes):
                clustering_data[com_id][class_i] = communities[com_id]["label_distribution"][class_i]
            clustering_data[com_id, :] /= clustering_data[com_id, :].sum()

        kmeans = KMeans(n_clusters=num_clients)
        kmeans.fit(clustering_data)

        clustering_labels = kmeans.labels_

        node_dict = {client_id: [] for client_id in range(num_clients)}
        
        local_graph_distribution = {i: [0]*G.num_classes for i in range(num_clients)}
        for com_id in range(num_communities):
            node_dict[clustering_labels[com_id]] += communities[com_id]["nodes"]
            local_graph_distribution[clustering_labels[com_id]].append(communities[com_id]["label_distribution"])

        for com_id in local_graph_distribution.keys():
            logging.info(local_graph_distribution[com_id])

    elif partition == "Metis+":
        logging.info("Conducting metis plus graph partition...")
        import pymetis as metis
        communities = {com_id: {"nodes":[], "num_nodes":0, "label_distribution":[0] * G.num_classes} 
                                for com_id in range(metis_num_coms)}
        n_cuts, membership = metis.part_graph(metis_num_coms, graph_nx)
        for com_id in range(metis_num_coms):
            com_indices = np.where(np.array(membership) == com_id)[0]
            com_indices = list(com_indices)
            communities[com_id]["nodes"] = com_indices
            communities[com_id]["num_nodes"] = len(com_indices)
            for node in communities[com_id]["nodes"]:
                label = copy.deepcopy(G.y[node])
                communities[com_id]["label_distribution"][label] += 1
        
        num_communities = len(communities)
        clustering_data = np.zeros(shape=(num_communities, G.num_classes))
        for com_id in communities.keys():
            for class_i in range(G.num_classes):
                clustering_data[com_id][class_i] = communities[com_id]["label_distribution"][class_i]
            clustering_data[com_id, :] /= clustering_data[com_id, :].sum()

        kmeans = KMeans(n_clusters=num_clients)
        kmeans.fit(clustering_data)

        clustering_labels = kmeans.labels_

        node_dict = {client_id: [] for client_id in range(num_clients)}
        
        for com_id in range(num_communities):
            node_dict[clustering_labels[com_id]] += communities[com_id]["nodes"]

    else:
        raise ValueError(f"No such partition method: '{partition}'.")

    assert sum([len(node_dict[i])
               for i in range(len(node_dict))]) == G.num_nodes
    

            
    subgraph_list = construct_subgraph_dict_from_node_dict(
        G=G,
        num_clients=num_clients,
        node_dict=node_dict,
        graph_nx=graph_nx,
        train=train,
        val=val,
        test=test,
    )
    for subgraph in subgraph_list:
        # node_homo, edge_homo = analysis_graph_structure_homo_hete_info(subgraph)
        _ = label_distribution(subgraph)
    
    G.num_samples = 0
    for subgraph in subgraph_list:
        G.num_samples += subgraph.num_samples
    return subgraph_list


def label_distribution(G):
    labels = G.y
    unique_labels, counts = torch.unique(labels, return_counts=True)

    label_distribution = dict(zip(unique_labels.tolist(), counts.tolist()))

    print(f"Label distribution: {label_distribution}")
    return label_distribution

def analysis_graph_structure_homo_hete_info(G):
    structure_homo_hete_label_info = {}
    structure_homo_hete_label_info["node_homophily"] = label_node_homogeneity(G)
    structure_homo_hete_label_info["edge_homophily"] = label_edge_homogeneity(G)
    print(
        f"homo_node: {structure_homo_hete_label_info['node_homophily']:.4f}\nhomo_edge: {structure_homo_hete_label_info['edge_homophily']:.4f}"
    )
    return (
        structure_homo_hete_label_info["node_homophily"],
        structure_homo_hete_label_info["edge_homophily"],
    )


def label_node_homogeneity(G):
    num_nodes = G.num_nodes
    homophily = 0
    for edge_u in range(num_nodes):
        hit = 0
        edge_v_list = G.edge_index[1][torch.where(G.edge_index[0] == edge_u)]
        if len(edge_v_list) != 0:
            for i in range(len(edge_v_list)):
                edge_v = edge_v_list[i]
                if G.y[edge_u] == G.y[edge_v]:
                    hit += 1
            homophily += hit / len(edge_v_list)
    homophily /= num_nodes
    return homophily

def node_homophily_details(G, num_classes):
    num_nodes = G.num_nodes
    homophily_class = {c: 0 for c in range(num_classes)}
    homophily_node = {nd: 0 for nd in range(num_nodes)}
    num_per_class = {c: 0 for c in range(num_classes)}
    homophily = 0.
    for edge_u in range(num_nodes):
        hit = 0
        edge_v_list = G.edge_index[1][torch.where(G.edge_index[0] == edge_u)]
        if len(edge_v_list) != 0:
            for i in range(len(edge_v_list)):
                edge_v = edge_v_list[i]
                if G.y[edge_u] == G.y[edge_v]:
                    hit += 1
            homophily = hit / len(edge_v_list)
        homophily_node[edge_u] = homophily
        homophily_class[G.y[edge_u].item()] += homophily
        num_per_class[G.y[edge_u].item()] += 1
    
    for c in range(num_classes):
        if num_per_class[c] == 0:
            continue
        homophily_class[c] /= num_per_class[c]

    return homophily_node, homophily_class

def node_degree_details(G, num_classes):
    num_nodes = G.num_nodes
    degree_class = {c: 0 for c in range(num_classes)}
    degree_node = {nd: 0 for nd in range(num_nodes)}
    num_per_class = {c: 0 for c in range(num_classes)}
    for edge_u in range(num_nodes):
        edge_v_list = G.edge_index[1][torch.where(G.edge_index[0] == edge_u)]
        degree_node[edge_u] = len(edge_v_list)
        degree_class[G.y[edge_u].item()] += len(edge_v_list)
        num_per_class[G.y[edge_u].item()] += 1
    
    for c in range(num_classes):
        if num_per_class[c] == 0:
            continue
        degree_class[c] /= num_per_class[c]

    return degree_node, degree_class


def label_edge_homogeneity(G):
    num_edges = G.num_edges
    homophily = 0
    for i in range(num_edges):
        if G.y[G.edge_index[0][i]] == G.y[G.edge_index[1][i]]:
            homophily += 1
    homophily /= num_edges
    return homophily



def construct_subgraph_dict_from_node_dict(num_clients, node_dict, G, graph_nx, train, val, test):
    logging.info(f"log from construct_subgraph_dict_from_node_dict: G.num_classes: {G.num_classes}")
    subgraph_list = []
    for client_id in range(num_clients):
        num_local_nodes = len(node_dict[client_id])
        logging.info(f"num_local_nodes for client {client_id}: {num_local_nodes}")
        
        train_idx = []
        val_idx = []
        test_idx = []
        class_i_idx_list = {}
        
        for idx in range(num_local_nodes):
            label = G.y[node_dict[client_id][idx]]
            if int(label) not in class_i_idx_list:
                class_i_idx_list[int(label)] = []
            class_i_idx_list[int(label)].append(idx)
        
        logging.info(class_i_idx_list.keys())
        logging.info([(c, len(class_i_idx_list[c])) for c in class_i_idx_list.keys()])
        
        for class_i in range(G.num_classes):
            if class_i not in class_i_idx_list:
                logging.info(f"client {client_id}, class {class_i}")
                continue
            local_node_idx = class_i_idx_list[class_i]
            random.shuffle(local_node_idx)

            num_local_nodes_class_i = len(local_node_idx)
            train_size = int(num_local_nodes_class_i * train)
            val_size = int(num_local_nodes_class_i * val)
            test_size = int(num_local_nodes_class_i * test) 

            train_idx += local_node_idx[:train_size]
            val_idx += local_node_idx[train_size: train_size + val_size]
            test_idx += local_node_idx[train_size + val_size:]

        assert len(train_idx) + len(val_idx) + len(test_idx) == num_local_nodes

        local_train_idx = idx_to_mask(train_idx, size=num_local_nodes)
        local_val_idx = idx_to_mask(val_idx, size=num_local_nodes)
        local_test_idx = idx_to_mask(test_idx, size=num_local_nodes)

        node_idx_map = {}
        edge_idx = []
        for idx in range(num_local_nodes):
            node_idx_map[node_dict[client_id][idx]] = idx
        edge_idx += [
            (node_idx_map[x[0]], node_idx_map[x[1]])
            for x in graph_nx.subgraph(node_dict[client_id]).edges
        ]
        edge_idx += [
            (node_idx_map[x[1]], node_idx_map[x[0]])
            for x in graph_nx.subgraph(node_dict[client_id]).edges
        ]

        edge_idx_tensor = torch.tensor(edge_idx, dtype=torch.long).T
        subgraph = Data(
            x=G.x[node_dict[client_id]],
            y=G.y[node_dict[client_id]],
            edge_index=edge_idx_tensor,
        )


        subgraph.train_idx = local_train_idx
        subgraph.val_idx = local_val_idx
        subgraph.test_idx = local_test_idx
        subgraph.num_samples = subgraph.num_nodes
        subgraph_list.append(subgraph)
        logging.info(
            "Client: {}\tTotal Nodes: {}\tTotal Edges: {}\tTrain Nodes: {}\tVal Nodes: {}\tTest Nodes\t{}".format(
                client_id + 1,
                subgraph.num_nodes,
                subgraph.num_edges,
                len(train_idx),
                len(val_idx),
                len(test_idx),
            )
        )

    return subgraph_list
