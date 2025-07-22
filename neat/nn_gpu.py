import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class Node:
    def __init__(self, id_, input_=False, output=False, layer=0):
        self.id = id_
        self.is_input = input_
        self.is_output = output
        self.layer = layer

class ConnectionGene:
    def __init__(self, in_node, out_node, innov_num, weight):
        self.in_node = in_node
        self.out_node = out_node
        self.innov_num = innov_num
        self.weight = weight
        self.enable = True

def reset_NN_class_state():
    NN.next_node_id = 0
    NN.innov_num_map = {}
    NN.next_innov_num = 0

class NN(nn.Module):
    next_node_id = 0
    innov_num_map = {}
    next_innov_num = 0

    def __init__(self, input_dim, output_dim, cloned=False):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nodes = []
        self.connections = []
        self.node_id_map = {}

        if not cloned:
            self._initialize(input_dim, output_dim)

        self._prepare_layers()

    def _initialize(self, input_dim, output_dim):
        for _ in range(input_dim):
            node = Node(NN.next_node_id, input_=True, layer=0)
            self.nodes.append(node)
            self.node_id_map[node.id] = node
            NN.next_node_id += 1

        for i in range(output_dim):
            node = Node(NN.next_node_id, output=True, layer=1)
            self.nodes.append(node)
            self.node_id_map[node.id] = node
            NN.next_node_id += 1

            for j in range(input_dim):
                in_id, out_id = j, node.id
                key = (in_id, out_id)
                if key not in NN.innov_num_map:
                    NN.innov_num_map[key] = NN.next_innov_num
                    NN.next_innov_num += 1
                innov = NN.innov_num_map[key]
                conn = ConnectionGene(self.node_id_map[in_id], node, innov, torch.randn(1))
                self.connections.append(conn)

    def _prepare_layers(self):
        self.layers = defaultdict(list)
        for node in self.nodes:
            self.layers[node.layer].append(node)

        self.max_layer = max(node.layer for node in self.nodes)
        self.layer_connections = []
        self.sparse_weights = []

        for l in range(self.max_layer):
            in_nodes = self.layers[l]
            out_nodes = self.layers[l + 1]

            in_ids = [n.id for n in in_nodes]
            out_ids = [n.id for n in out_nodes]
            in_id_to_idx = {nid: i for i, nid in enumerate(in_ids)}
            out_id_to_idx = {nid: i for i, nid in enumerate(out_ids)}

            indices = []
            values = []

            for conn in self.connections:
                if not conn.enable:
                    continue
                if conn.in_node.id in in_id_to_idx and conn.out_node.id in out_id_to_idx:
                    i = out_id_to_idx[conn.out_node.id]
                    j = in_id_to_idx[conn.in_node.id]
                    indices.append([i, j])
                    values.append(conn.weight.item())

            if indices:
                idx_tensor = torch.tensor(indices, dtype=torch.long).t()  # shape (2, N)
                val_tensor = torch.tensor(values, dtype=torch.float32)
                shape = (len(out_ids), len(in_ids))
                W_sparse = torch.sparse_coo_tensor(idx_tensor, val_tensor, size=shape)
            else:
                shape = (len(out_ids), len(in_ids))
                W_sparse = torch.sparse_coo_tensor(
                    torch.zeros((2, 0), dtype=torch.long),
                    torch.zeros(0),
                    size=shape
                )

            self.sparse_weights.append(W_sparse)
            self.layer_connections.append((in_ids, out_ids))

    def forward(self, x):
        A = x  # shape: [batch_size, input_dim]

        for W_sparse, (in_ids, out_ids) in zip(self.sparse_weights, self.layer_connections):
            W = W_sparse.to(x.device)

            A = torch.matmul(A, W.t())  # shape: [batch_size, len(out_ids)]
            A = torch.sigmoid(A)

        return A  # last layer output

    def clone(self):
        new_net = NN(self.input_dim, self.output_dim, cloned=True)

        # Clone nodes
        new_net.nodes = []
        new_net.node_id_map = {}
        for node in self.nodes:
            clone_node = Node(node.id, node.is_input, node.is_output, node.layer)
            new_net.nodes.append(clone_node)
            new_net.node_id_map[clone_node.id] = clone_node

        # Clone connections
        new_net.connections = []
        for conn in self.connections:
            in_node = new_net.node_id_map[conn.in_node.id]
            out_node = new_net.node_id_map[conn.out_node.id]
            new_conn = ConnectionGene(
                in_node, out_node, conn.innov_num, conn.weight.clone().detach()
            )
            new_conn.enable = conn.enable
            new_net.connections.append(new_conn)

        new_net._prepare_layers()
        return new_net

    def to(self, device):
        self.sparse_weights = [W.to(device) for W in self.sparse_weights]
        return self
