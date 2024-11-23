import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, add_self_loops, dense_to_sparse
from torch_geometric.data import Data
import sklearn
import logging


def accuracy(pred, ground_truth, by_class=False):
    y_hat = pred.max(1)[1]
    correct = (y_hat == ground_truth).nonzero().shape[0]
    acc = correct / ground_truth.shape[0]

    if by_class:
        unique_classes = ground_truth.unique()
        class_accuracies = {}
        for cls in unique_classes:
            class_indices = (ground_truth == cls)
            correct_predictions = (y_hat[class_indices] == cls).sum().item()
            total_samples = class_indices.sum().item()
            if total_samples > 0:
                class_accuracy = correct_predictions / total_samples * 100  # Convert to percentage
            else:
                class_accuracy = 0.0  # If no samples for this class, accuracy is 0
            class_accuracies[int(cls)] = class_accuracy

        return class_accuracies, acc * 100
    else:
        return acc * 100


def accuracy_per_class(pred: torch.Tensor, ground_truth: torch.Tensor):
    """
    Calculate accuracy for each class in the predictions against the ground truth labels.
    
    Args:
        pred (torch.Tensor): A 2D tensor where each row contains the predicted scores for each class.
        ground_truth (torch.Tensor): A 1D tensor containing the true class labels.

    Returns:
        Dict[int, float]: A dictionary where keys are class indices and values are the accuracy as percentages.
    """
    # Ensure the input tensors are not empty
    if pred.numel() == 0 or ground_truth.numel() == 0:
        raise ValueError("Input tensors cannot be empty.")

    # Get the predicted class indices by taking the argmax across classes
    y_hat = pred.argmax(dim=1)
    
    # Get unique classes in the ground truth
    unique_classes = ground_truth.unique()
    class_accuracies = {}

    # Calculate accuracy for each class
    for cls in unique_classes:
        # Get indices for the current class
        class_indices = (ground_truth == cls)
        
        # Get the number of correct predictions for this class
        correct_predictions = (y_hat[class_indices] == cls).sum().item()
        
        # Get the total number of samples for this class
        total_samples = class_indices.sum().item()
        
        # Calculate accuracy for this class
        if total_samples > 0:
            class_accuracy = correct_predictions / total_samples * 100  # Convert to percentage
        else:
            class_accuracy = 0.0  # If no samples for this class, accuracy is 0

        # Store the accuracy in the dictionary
        class_accuracies[int(cls)] = class_accuracy
    
    return class_accuracies



def f1_score(pred, ground_truth, num_classes):
    y_hat = pred.max(1)[1]
    # f1_micro = sklearn.metrics.f1_score(ground_truth, y_hat, average="micro")
    labels = [i for i in range(num_classes)]
    f1_class = sklearn.metrics.f1_score(ground_truth.cpu().numpy(), y_hat.cpu().numpy(), average=None, labels=labels)
    f1_macro = sklearn.metrics.f1_score(ground_truth.cpu().numpy(), y_hat.cpu().numpy(), average="macro", labels=labels)
    return f1_class, f1_macro


def student_loss(s_logit, t_logit, return_t_logits=False, method="kl"):
    """Kl/ L1 Loss for student"""
    print_logits = False
    if method == "l1":
        loss_fn = F.l1_loss
        loss = loss_fn(s_logit, t_logit.detach())
    elif method == "kl":
        loss_fn = F.kl_div
        s_logit = F.log_softmax(s_logit, dim=1)
        t_logit = F.softmax(t_logit, dim=1)
        loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
    else:
        raise ValueError(method)

    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss


class DiversityLoss(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))

    
def construct_graph(node_logits, adj_logits, k=5):
    adjacency_matrix = torch.zeros_like(adj_logits)
    topk_values, topk_indices = torch.topk(adj_logits, k=k, dim=1)
    for i in range(node_logits.shape[0]):
        adjacency_matrix[i, topk_indices[i]] = 1
    adjacency_matrix = adjacency_matrix + adjacency_matrix.t()
    adjacency_matrix[adjacency_matrix > 1] = 1
    adjacency_matrix.fill_diagonal_(1)
    edge = adjacency_matrix.long()
    edge_index, _ = dense_to_sparse(edge)
    edge_index = add_self_loops(edge_index)[0]
    data = Data(x=node_logits, edge_index=edge_index)
    return data   
    

def random_walk_with_matrix(T, walk_length, start):
    current_node = start
    walk = [current_node]
    for _ in range(walk_length - 1):
        probabilities = F.softmax(T[current_node, :], dim=0)
        probabilities /= torch.sum(probabilities)
        next_node = torch.multinomial(probabilities, 1).item()
        walk.append(next_node)
        current_node = next_node
    return walk




def cal_topo_emb(edge_index, num_nodes, max_walk_length):
    A = to_dense_adj(add_self_loops(edge_index)[0], max_num_nodes=num_nodes).squeeze()
    D = torch.diag(torch.sum(A, dim=1))
    torch.backends.cuda.preferred_linalg_library("magma")
    T = A * torch.pinverse(D)
    result_each_length = []
    logging.info("after calculate T")
    for i in range(1, max_walk_length+1):    
        result_per_node = []
        for start in range(num_nodes):
            result_walk = random_walk_with_matrix(T, i, start)
            result_per_node.append(torch.tensor(result_walk).view(1,-1))
        result_each_length.append(torch.vstack(result_per_node))
    topo_emb = torch.hstack(result_each_length)
    return topo_emb    



def cal_lp_emb(edge_index, num_nodes, x, hop=1):
    A = to_dense_adj(add_self_loops(edge_index)[0], max_num_nodes=num_nodes).squeeze()
    D = torch.diag(torch.sum(A, dim=1))
    T = torch.pinverse(D) @ A
    
    for i in range(hop):    
        x = torch.matmul(T, x)
    
    return x