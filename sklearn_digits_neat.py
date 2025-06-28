from sklearn.datasets import load_digits
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

digits = load_digits()

data_tensor = torch.tensor(digits.data, dtype=torch.float32)
target_tensor = F.one_hot(torch.tensor(digits.target), num_classes=10).float()

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    data_tensor, target_tensor, test_size=0.2, random_state=42, shuffle=True
)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Batch is the full size because there is no backpropogation
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))



# NEAT Classes

from collections import deque, defaultdict
import random

torch.manual_seed(42)

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
    node_count = 0
    
    # key: (in, out) 
    # value: resulting node
    # index: the innov_num 
    global_connections = {}
    
    def __init__(self, input_dim, output_dim, cloned=False):
        super(NN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.nodes = [] # Nodes objects in this specific NN
        self.connections = [] # Connections objects in this specific NN

        # Cloned models should not be inited!
        if cloned:
            return
        else:
            # When init a new model, the node nums and innov should be the same as any other new initialized model
            # This branch is for initial population models
            # Initalize a fully connected NN with no hidden layers
            for i in range(input_dim):
                if i >= NN.node_count:
                    NN.node_count += 1
                self.nodes.append(Node(i, True))
    
            for i in range(output_dim):
                node_index = i + input_dim
                if node_index >= NN.node_count:
                    NN.node_count += 1
                self.nodes.append(Node(node_index, False, True))

                for in_id in range(input_dim):
                    in_out_tuple = (in_id, node_index)
                    if in_out_tuple not in list(NN.global_connections.keys()):
                        NN.global_connections.update({in_out_tuple: None}) # No resulting node until it is split for the first time
                    innov_num = list(NN.global_connections.keys()).index(in_out_tuple)

                    self.connections.append(ConnectionGene(self.nodes[in_id], self.nodes[node_index], innov_num, torch.randn(1)))
                    self.nodes[node_index].num_incoming_connections += 1 # Each output starts off fully connected to input

    def clone(self):
        # Create a new NN instance with same input/output dims
        new_nn = NN(self.input_dim, self.output_dim, True)
        
        # Deep copy nodes
        new_nn.nodes = []
        for node in self.nodes:
            new_node = Node(node.id, node.is_input, node.is_output)
            new_node.num_incoming_connections = node.num_incoming_connections
            new_nn.nodes.append(new_node)

        # Deep copy connections
        new_nn.connections = []
        for conn in self.connections:
            # Find the corresponding new nodes by id
            in_node = next(n for n in new_nn.nodes if n.id == conn.in_node.id)
            out_node = next(n for n in new_nn.nodes if n.id == conn.out_node.id)

            new_conn = ConnectionGene(in_node, out_node, conn.innov_num, conn.weight.clone().detach())
            new_conn.enable = conn.enable
            new_nn.connections.append(new_conn)

        return new_nn

    def forward(self, x: torch.Tensor):
        if x.shape[1] != self.input_dim:
            raise ValueError("Input dim is not correct")
        
        batch_size = x.shape[0]
    
        # Create batch versions of node values and received counts and reset them to zero
        for node in self.nodes:
            node.val = torch.zeros(batch_size)
            node.received = torch.zeros(batch_size, dtype=torch.int)
    
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
    
            for conn in self.connections:
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
                        out_node.val = torch.relu(out_node.val)
                    queue.append(out_node)
    
        # Collect logits from output nodes
        output_vals = [node.val for node in self.nodes if node.is_output]
        logits = torch.stack(output_vals, dim=1)  # shape: (batch_size, num_outputs)
        return logits

    # I think in the paper perturbation and modification are the same but i did different
    def weight_perturbation(self, quiet=False):
        rand_conn_id = torch.randint(0, len(self.connections), (1,)).item()
        
        mean = 0.0
        std_dev = 0.1
        
        noise = torch.randn_like(self.connections[rand_conn_id].weight) * std_dev + mean
        self.connections[rand_conn_id].weight += noise
        
        if not quiet:
            print(f'connection {rand_conn_id} weight perturbated to {self.connections[rand_conn_id].weight.item():.2f}')
        
    def weight_modification(self, quiet=False):
        rand_conn_id = torch.randint(0, len(self.connections), (1,)).item()

        self.connections[rand_conn_id].weight = torch.randn(1)
        
        if not quiet:
            print(f'connection {rand_conn_id} weight modified to {self.connections[rand_conn_id].weight.item():.2f}')

    def add_connection(self, quiet=False):
        max_attempts = 100  # prevent infinite loop
        for _ in range(max_attempts):
            rand_node_in = self.nodes[torch.randint(0, len(self.nodes), (1,)).item()]
        
            if rand_node_in.is_output:
                continue
        
            rand_node_out = self.nodes[torch.randint(0, len(self.nodes), (1,)).item()]
        
            if rand_node_out == rand_node_in or rand_node_out.is_input:
                continue
        
            # Skip if connection already exists
            # Basically no add connection will work in the beginning because its fully connected
            if any(conn.in_node == rand_node_in and conn.out_node == rand_node_out for conn in self.connections):
                continue
        
            # Optional: skip if this would form a cycle
            # if self.creates_cycle(rand_node_in, rand_node_out):
            #     continue
            
            if (rand_node_in.id, rand_node_out.id) not in list(NN.global_connections.keys()):
                NN.global_connections.update({(rand_node_in.id, rand_node_out.id): 0})
            innov_num = list(NN.global_connections.keys()).index((rand_node_in.id, rand_node_out.id))
            
            conn = ConnectionGene(rand_node_in, rand_node_out, innov_num, torch.randn(1))
            self.connections.append(conn)
            rand_node_out.num_incoming_connections += 1
            
            if not quiet:
                print(f"Connection created from node {rand_node_in.id} to node {rand_node_out.id} (innovation #{innov_num})")

            return
            
        if not quiet:
            print("Failed to add connection after max attempts.")

    def add_node(self, quiet=False):            
        rand_conn_id = torch.randint(0, len(self.connections), (1,)).item()

        while not self.connections[rand_conn_id].enable:
            rand_conn_id = torch.randint(0, len(self.connections), (1,)).item()

        # Splits an existing connection by adding a node
        self.connections[rand_conn_id].enable = False

        if NN.global_connections[(self.connections[rand_conn_id].in_node.id, self.connections[rand_conn_id].out_node.id)] is not None:
            new_node_id = NN.global_connections[(self.connections[rand_conn_id].in_node.id, self.connections[rand_conn_id].out_node.id)]
        else:
            new_node_id = NN.node_count
            NN.global_connections[(self.connections[rand_conn_id].in_node.id, self.connections[rand_conn_id].out_node.id)] = new_node_id
            NN.node_count += 1

        new_node = Node(new_node_id)
        new_node.num_incoming_connections += 1
        self.nodes.append(new_node)

        in_out_tuple = (self.connections[rand_conn_id].in_node.id, new_node.id)
        if in_out_tuple not in list(NN.global_connections.keys()):
            NN.global_connections.update({in_out_tuple: None}) # No resulting node until it is split for the first time
        innov_num = list(NN.global_connections.keys()).index(in_out_tuple)
        
        conn = ConnectionGene(self.connections[rand_conn_id].in_node, new_node, innov_num, torch.randn(1))
        self.connections.append(conn)

        in_out_tuple = (new_node.id, self.connections[rand_conn_id].out_node.id)
        if in_out_tuple not in list(NN.global_connections.keys()):
            NN.global_connections.update({in_out_tuple: None}) # No resulting node until it is split for the first time
        innov_num = list(NN.global_connections.keys()).index(in_out_tuple)
        
        conn = ConnectionGene(new_node, self.connections[rand_conn_id].out_node, innov_num, torch.randn(1))
        self.connections.append(conn)
        
        if not quiet:
            print(f'connection {rand_conn_id} split')

    def toggle_connection(self, quiet=False):
        rand_conn_id = torch.randint(0, len(self.connections), (1,)).item()

        self.connections[rand_conn_id].enable = not self.connections[rand_conn_id].enable

        if not quiet:
            print(f'connection {rand_conn_id} toggled to {self.connections[rand_conn_id].enable}')

    def mutate(self, quiet=False):
        # For this experiment i use 70 weight perturbation, 20 weight mutation, 30 add connection, 3 add node, 2 toggle
        # Each one is chosen independently of each other
        # Does not include crossover. That cannot be done by itself
        new_model = self.clone()

        # Do mutations on new_model
        if random.random() < .7:
            new_model.weight_perturbation(quiet)
        if random.random() < .2:
            new_model.weight_modification(quiet)
        if random.random() < .3:
            new_model.add_connection(quiet)
        if random.random() < .03:
            new_model.add_node(quiet)
        if random.random() < .02:
            new_model.toggle_connection(quiet)
    
        return new_model
        
def crossover(info1, info2):
    # Equal Fitness
    # Might not implement this rn because it doesnt happen that much
    # if info1['loss'] == info2['loss']:
    #     print("Same loss")
    #     return

    # Find fitter model
    fit_model, less_fit_model = (info2['model'], info1['model']) if info1['loss'] > info2['loss'] else (info1['model'], info2['model'])

    # New model starts off as clone of more fit
    new_model = fit_model.clone()

    less_fit_conns = {conn.innov_num: conn for conn in less_fit_model.connections}

    for i in new_model.connections:
        if i.innov_num in less_fit_conns:
            # Randomly decides to inherit from less_fit if there is a matching connection
            if random.random() < 0.5:
                other_conn = less_fit_conns[i.innov_num]
                i.enable = other_conn.enable

                # Idk this is in the paper
                if not i.enable or not other_conn.enable:
                    i.enable = random.random() < 0.25  
                else:     
                    i.weight = other_conn.weight

    return new_model

def measure_compatibility(genome1, genome2, c1, c2, c3, delta_thresh):
    genome1_conns = {i.innov_num: i.weight for i in genome1.connections}
    genome2_conns = {i.innov_num: i.weight for i in genome2.connections}

    innovs1 = set(genome1_conns.keys())
    innovs2 = set(genome2_conns.keys())

    max_innov1 = max(innovs1) if innovs1 else 0
    max_innov2 = max(innovs2) if innovs2 else 0
    max_innov = max(max_innov1, max_innov2)

    # Matching genes: innovation numbers in both genomes
    matching = innovs1.intersection(innovs2)
    # Calculate average weight difference for matching genes
    if matching:
        weight_diff = sum(abs(genome1_conns[i] - genome2_conns[i]) for i in matching) / len(matching)
    else:
        weight_diff = 0

    # Excess genes: genes whose innovation number is greater than max innovation number of other genome
    excess = 0
    for innov in innovs1:
        if innov > max_innov2:
            excess += 1
    for innov in innovs2:
        if innov > max_innov1:
            excess += 1

    # Disjoint genes: genes that do not match and are not excess
    disjoint = (len(innovs1 - innovs2) + len(innovs2 - innovs1)) - excess

    # Normalization factor N
    N = max(len(genome1_conns), len(genome2_conns))
    if N < 20:
        N = 1  # as per original NEAT paper for small genomes

    delta = (c1 * excess / N) + (c2 * disjoint / N) + (c3 * weight_diff)
        
    return True if delta < delta_thresh else False

# Init stuff - Should be pretty good

# Hyperparameters
population_size = 50
epochs = 3000
input_dim = 64
output_dim = 10
top_k = 0.6 # The percentage of genomes to keep for reproduction
crossover_percent = 0.5

# hyperparameters for measuring compatibility from https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf
c1 = 1.0
c2 = 1.0
c3 = 3.0
delta_thresh = 4.0

# Using list of lists
# Dead species will not be kept track of. There will be no empty list
population = []

# Init first model
new_model = {"model": NN(input_dim, output_dim), "loss": float('inf'), "fitness": -float('inf')}
population.append([new_model])

for _ in range(population_size - 1):
    new_model = {"model": NN(input_dim, output_dim), "loss": float('inf'), "fitness": -float('inf')}
    
    added = False
    for idx, species in enumerate(population):
        if measure_compatibility(new_model['model'], species[0]['model'], c1, c2, c3, delta_thresh):
            population[idx].append(new_model)
            added = True
            break
    if not added:
        # New species created
        population.append([new_model])

loss_fn = nn.CrossEntropyLoss()

# "Training loop"

import math

for epoch in range(epochs):
    print(f"epoch: {epoch}")
    
    for species in population:
        
        with torch.no_grad():
            for model_info in species:
                model = model_info["model"]
                total_loss = 0.0
                total_samples = 0
    
                for data_batch, label_batch in train_loader:
                    output = model(data_batch)
                    loss = loss_fn(output, label_batch)
                    total_loss += loss.item() * data_batch.size(0)
                    total_samples += data_batch.size(0)
                
                model_info["loss"] = total_loss / total_samples

    flattened_population = []

    for species in population:
        for genome in species:
            flattened_population.append(genome)
            
    ranked_models = sorted([model_info for model_info in flattened_population], key=lambda x: x["loss"])

    print(f"top model loss: {ranked_models[0]['loss']:.2f}")

    # Fitness sharing
    for species in population:
        species_size = len(species)
        for genome in species:
            raw_fitness = 1 / (1 + genome['loss'])
            genome['fitness'] = raw_fitness / species_size

    # This is just a list not a list of lists
    new_population = []

    for species in population:
        offspring = []

        ranked_models = sorted([model_info for model_info in species], key=lambda x: x["fitness"], reverse=True)
        parents = [model_info for model_info in ranked_models[:math.ceil(top_k * len(ranked_models))]]

        for i in range(math.ceil(crossover_percent * len(ranked_models))):
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            child = crossover(p1, p2)
            offspring.append({"model": child, "loss": float('inf'), "fitness": -float('inf')})
    
        while len(offspring) != len(ranked_models):
            offspring.append({"model": random.choice(parents)['model'].mutate(True), "loss": float('inf'), "fitness": -float('inf')})
            
        new_population.extend(offspring)

    # Redivide into species
    new_population_divided = []

    for model in new_population:    
        # First model
        if len(new_population_divided) == 0:
            new_population_divided.append([model])
        else:
            added = False
            for idx, species in enumerate(new_population_divided):
                if measure_compatibility(model['model'], species[0]['model'], c1, c2, c3, delta_thresh):
                    new_population_divided[idx].append(model)
                    added = True
                    break
            if not added:
                # New species created
                new_population_divided.append([model])
                    
    population = new_population_divided
    
    print(len(population))


# Get best model
for species in population:
        
        with torch.no_grad():
            for model_info in species:
                model = model_info["model"]
                total_loss = 0.0
                total_samples = 0
    
                for data_batch, label_batch in train_loader:
                    output = model(data_batch)
                    loss = loss_fn(output, label_batch)
                    total_loss += loss.item() * data_batch.size(0)
                    total_samples += data_batch.size(0)
                
                model_info["loss"] = total_loss / total_samples

flattened_population = []

for species in population:
    for genome in species:
        flattened_population.append(genome)
        
ranked_models = sorted([model_info for model_info in flattened_population], key=lambda x: x["loss"])

model = ranked_models[0]['model']

# Evaulate
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, one_hot_labels in test_loader:
        data, one_hot_labels = data, one_hot_labels

        outputs = model(data)  # logits
        predicted = torch.argmax(outputs, dim=1)  # class indices
        true_labels = torch.argmax(one_hot_labels, dim=1)  # class indices from one-hot

        correct += (predicted == true_labels).sum().item()
        total += true_labels.size(0)

accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")