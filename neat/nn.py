import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, defaultdict

class Node:
    def __init__(self, id_, input_=False, output=False):
        self.id = id_
        self.is_input = input_
        self.is_output = output

        self.val = None # Tensors
        
        self.num_incoming_connections = 0 
        self.received = None # Keep track of nodes received before applying activation function


# --------------------------------------------------------------------------------------------------------


class ConnectionGene:
    def __init__(self, in_node, out_node, innov_num, weight):
        self.in_node = in_node # Nodes not node id
        self.out_node = out_node
        
        self.innov_num = innov_num
        
        self.weight = weight # Weights are tensors
        self.enable = True # If node is disabled, it CAN be reenabled


# --------------------------------------------------------------------------------------------------------


class NN(nn.Module):

    # For assigning node ids to new nodes
    next_node_id = 0
    
    # key: (in, out) 
    # value: resulting node
    resulting_node_map = {}

    # key: (in, out) 
    # value: the innov_num 
    innov_num_map = {}
    next_innov_num = 0
    
    def __init__(self, input_dim, output_dim, cloned=False):
        super(NN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.nodes = [] # Nodes objects in this specific NN
        
        self.connections_by_id = {} # Connections objects in this specific NN | in node id is key for forward lookup
        self.connections = [] # All connections for mutating
        
        # Cloned models should not be inited!
        if cloned:
            return
        else:
            # When init a new model, the node nums and innov should be the same as any other new initialized model
            # This branch is for initial population models
            # Initalize a fully connected NN with no hidden layers
            for i in range(input_dim):
                if i >= NN.next_node_id:
                    NN.next_node_id += 1
                self.nodes.append(Node(i, True))
    
            for i in range(output_dim):
                node_index = i + input_dim
                if node_index >= NN.next_node_id:
                    NN.next_node_id += 1
                self.nodes.append(Node(node_index, False, True))

                for in_id in range(input_dim):
                    in_out_tuple = (in_id, node_index)
                    
                    if in_out_tuple not in NN.resulting_node_map:
                        NN.resulting_node_map.update({in_out_tuple: None}) # No resulting node until it is split for the first time
                        NN.innov_num_map.update({in_out_tuple: NN.next_innov_num}) # No resulting node until it is split for the first time
                        NN.next_innov_num += 1
                        
                    innov_num = NN.innov_num_map[in_out_tuple]

                    conn = ConnectionGene(self.nodes[in_id], self.nodes[node_index], innov_num, torch.randn(1))
                    
                    if in_id not in self.connections_by_id:
                        self.connections_by_id[in_id] = [conn]
                    else:
                        self.connections_by_id[in_id].append(conn)

                    self.connections.append(conn)
                    
                    self.nodes[node_index].num_incoming_connections += 1 # Each output starts off fully connected to input
                            
    def clone(self):
        # Create a new NN instance with same input/output dims
        new_nn = NN(self.input_dim, self.output_dim, cloned=True)
    
        # Deep copy nodes
        new_nn.nodes = []
        id_to_node = {}
        for node in self.nodes:
            new_node = Node(node.id, node.is_input, node.is_output)
            new_node.num_incoming_connections = node.num_incoming_connections
            new_nn.nodes.append(new_node)
            id_to_node[node.id] = new_node
    
        # Deep copy connections
        new_nn.connections = []

        new_nn.connections_by_id = {}

        for conn_list in self.connections_by_id.values():  # each value is a list of connections
            for conn in conn_list:
                in_node = id_to_node[conn.in_node.id]
                out_node = id_to_node[conn.out_node.id]
        
                new_conn = ConnectionGene(
                    in_node, out_node, conn.innov_num, conn.weight.clone().detach()
                )
                new_conn.enable = conn.enable

                new_nn.connections.append(new_conn)

                if in_node.id not in new_nn.connections_by_id:
                    new_nn.connections_by_id[in_node.id] = [new_conn]
                else:
                    new_nn.connections_by_id[in_node.id].append(new_conn)
        
        return new_nn

    def forward(self, x: torch.Tensor):
        # Flatten square images
        x = x.view(x.size(0), -1)

        if x.shape[1] != self.input_dim:
            raise ValueError("Input dim is not correct")
        
        batch_size = x.shape[0]

        device = x.device


        # Create batch versions of node values and received counts and reset them to zero
        for node in self.nodes:
            node.val = torch.zeros(batch_size, device=device)
            node.received = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
        # Set input values
        for idx in range(self.input_dim):
            self.nodes[idx].val = x[:, idx]

        # Start with nodes whose incoming connections are already satisfied AKA input nodes
        queue = deque()
        for node in self.nodes:
            if node.num_incoming_connections == 0:
                queue.append(node)
    
        while queue:
            curr_node = queue.popleft()

            
            for conn in self.connections_by_id.get(curr_node.id, []):
                if not conn.enable:
                    continue
                if conn.in_node != curr_node:
                    continue

                out_node = conn.out_node
                out_node.val += (curr_node.val * conn.weight)

                out_node.received += 1
    
                # Only enqueue if all inputs are received
                # Note: vectorized check — adds node to queue if all samples are ready
                if (out_node.received == out_node.num_incoming_connections).all():
                    if not out_node.is_output:
                        out_node.val = torch.sigmoid(out_node.val)
                        # Out nodes are never added here and sigmoid is not applied to them
                        queue.append(out_node)
    
        # Collect logits from output nodes
        output_vals = [node.val for node in self.nodes if node.is_output]
        logits = torch.stack(output_vals, dim=1)  # shape: (batch_size, num_outputs)
        return logits

    def to(self, device):
        for node in self.nodes:
            if node.val is not None:
                node.val = node.val.to(device)
            if node.received is not None:
                node.received = node.received.to(device)
        for conn in self.connections:
            conn.weight = conn.weight.to(device)
        return self

def reset_NN_class_state():
    NN.next_node_id = 0
    NN.resulting_node_map = {}
    NN.innov_num_map = {}
    NN.next_innov_num = 0