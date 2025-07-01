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
                # out_node.val += (curr_node.val)

                out_node.received += 1
    
                # Only enqueue if all inputs are received
                # Note: vectorized check — adds node to queue if all samples are ready
                if (out_node.received == out_node.num_incoming_connections).all():
                    if not out_node.is_output:
                        out_node.val = torch.sigmoid(out_node.val)
                    if not out_node.is_output:
                        # Out nodes are never added here and sigmoid is not applied to them
                        queue.append(out_node)
    
        # Collect logits from output nodes
        output_vals = [node.val for node in self.nodes if node.is_output]
        logits = torch.stack(output_vals, dim=1)  # shape: (batch_size, num_outputs)
        return logits

    # I think in the paper perturbation and modification are the same but i did different
    def weight_perturbation(self, quiet=False):
        rand_conn = random.choice(self.connections)
        
        mean = 0.0
        std_dev = 0.5
        
        noise = torch.randn_like(rand_conn.weight) * std_dev + mean
        rand_conn.weight += noise
        
        if not quiet:
            print(f'connection {rand_conn.innov_num} (in this nn) weight perturbated to {rand_conn.weight.item():.2f}')
        
    def weight_modification(self, quiet=False):
        rand_conn = random.choice(self.connections)

        rand_conn.weight = torch.randn(1)
        
        if not quiet:
            print(f'connection {rand_conn.innov_num} (in this nn) weight modified to {rand_conn.weight.item():.2f}')

    def creates_cycle(self, source, target):
        """Returns True if adding an edge from `source` to `target` would create a cycle."""
        visited = set()
    
        def dfs(node):
            if node.id in visited:
                return False
            if node == source:
                return True  # Found a path back to source — would create cycle
            visited.add(node.id)
            for conn in self.connections:
                if conn.enable and conn.in_node == node:
                    if dfs(conn.out_node):
                        return True
            return False
    
        return dfs(target)

    
    def add_connection(self, quiet=False):
        max_attempts = 1000  # prevent infinite loop
        for _ in range(max_attempts):
            rand_node_in = random.choice(self.nodes)

            if rand_node_in.is_output:
                continue
        
            rand_node_out = random.choice(self.nodes)
            
            if rand_node_out == rand_node_in or rand_node_out.is_input:
                continue
        
            # Skip if connection already exists
            # Basically no add connection will work in the beginning because its fully connected
            if any(conn.in_node == rand_node_in and conn.out_node == rand_node_out for conn in self.connections):
                continue

            # Prevents acylic
            if self.creates_cycle(rand_node_in, rand_node_out):
                continue

            in_out_tuple = (rand_node_in.id, rand_node_out.id)

            if in_out_tuple not in NN.resulting_node_map:
                NN.resulting_node_map.update({in_out_tuple: None}) # No resulting node until it is split for the first time
                NN.innov_num_map.update({in_out_tuple: NN.next_innov_num}) # No resulting node until it is split for the first time
                NN.next_innov_num += 1
    
            innov_num = NN.innov_num_map[in_out_tuple]
    
            conn = ConnectionGene(rand_node_in, rand_node_out, innov_num, torch.randn(1))
            
            if rand_node_in.id not in self.connections_by_id:
                self.connections_by_id[rand_node_in.id ] = [conn]
            else:
                self.connections_by_id[rand_node_in.id ].append(conn)
    
            self.connections.append(conn)
        
            rand_node_out.num_incoming_connections += 1
            
            if not quiet:
                print(f"Connection created from node {rand_node_in.id} to node {rand_node_out.id} (innovation #{innov_num})")

            return
            
        if not quiet:
            print("Failed to add connection after max attempts.")

    def add_node(self, quiet=False):   
        enabled_conn_ids = [i for i, c in enumerate(self.connections) if c.enable]
        
        if not enabled_conn_ids:
            if not quiet:
                print("No enabled connections to split.")
            return
            
        rand_conn_id = random.choice(enabled_conn_ids)

        # Splits an existing connection by adding a node
        self.connections[rand_conn_id].enable = False

        old_in_out_pair = (self.connections[rand_conn_id].in_node.id, self.connections[rand_conn_id].out_node.id)
        # Since this connection already exists, it should be in the map. Whether it is none or not is decided
        if NN.resulting_node_map[old_in_out_pair] is not None:
            new_node_id = NN.resulting_node_map[old_in_out_pair]
        else:
            new_node_id = NN.next_node_id
            NN.resulting_node_map[old_in_out_pair] = new_node_id
            NN.next_node_id += 1

        new_node = Node(new_node_id)
        self.nodes.append(new_node)

        in_out_tuple = (self.connections[rand_conn_id].in_node.id, new_node.id)

        if in_out_tuple not in NN.resulting_node_map:
            NN.resulting_node_map.update({in_out_tuple: None}) # No resulting node until it is split for the first time
            NN.innov_num_map.update({in_out_tuple: NN.next_innov_num}) # No resulting node until it is split for the first time
            NN.next_innov_num += 1

        innov_num = NN.innov_num_map[in_out_tuple]

        conn = ConnectionGene(self.connections[rand_conn_id].in_node, new_node, innov_num, torch.randn(1))
        
        if self.connections[rand_conn_id].in_node.id not in self.connections_by_id:
            self.connections_by_id[self.connections[rand_conn_id].in_node.id ] = [conn]
        else:
            self.connections_by_id[self.connections[rand_conn_id].in_node.id ].append(conn)

        self.connections.append(conn)

        in_out_tuple = (new_node.id, self.connections[rand_conn_id].out_node.id)

        if in_out_tuple not in NN.resulting_node_map:
            NN.resulting_node_map.update({in_out_tuple: None}) # No resulting node until it is split for the first time
            NN.innov_num_map.update({in_out_tuple: NN.next_innov_num}) # No resulting node until it is split for the first time
            NN.next_innov_num += 1
 
        innov_num = NN.innov_num_map[in_out_tuple]

        conn = ConnectionGene(new_node, self.connections[rand_conn_id].out_node, innov_num, torch.randn(1))
        
        if new_node.id not in self.connections_by_id:
            self.connections_by_id[new_node.id] = [conn]
        else:
            self.connections_by_id[new_node.id].append(conn)

        self.connections.append(conn)
        new_node.num_incoming_connections += 1
  
        if not quiet:
            print(f'connection {rand_conn_id} split')

    def toggle_connection(self, quiet=False):
        rand_conn = random.choice(self.connections)

        rand_conn.enable = not rand_conn.enable
        
        if not rand_conn.enable:
            rand_conn.out_node.num_incoming_connections -= 1
        else:
            rand_conn.out_node.num_incoming_connections += 1

        if not quiet:
            print(f'connection {rand_conn.innov_num} (in this nn) toggled to {rand_conn.enable}')

    def mutate(self, quiet=False):
        # Each one is chosen independently of each other
        # Does not include crossover. That cannot be done by itself
        new_model = self.clone()

        # Do mutations on new_model
        if random.random() < .8:
            new_model.weight_perturbation(quiet)
        if random.random() < .2:
            new_model.weight_modification(quiet)
        if random.random() < .1:
            new_model.add_connection(quiet)
        if random.random() < .05:
            new_model.add_node(quiet)
        if random.random() < .02:
            new_model.toggle_connection(quiet)
    
        return new_model

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