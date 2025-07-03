"""
Functions on a genome

TO DO: Move mutate stuff here from cppn file and nn file
"""
import random
import torch
from neat.nn import *

# I think in the paper perturbation and modification are the same but i did different
def weight_perturbation(model, quiet=False):
    rand_conn = random.choice(model.connections)
    
    mean = 0.0
    std_dev = 0.5
    
    noise = torch.randn_like(rand_conn.weight) * std_dev + mean
    rand_conn.weight += noise
    
    if not quiet:
        print(f'connection {rand_conn.innov_num} (in this nn) weight perturbated to {rand_conn.weight.item():.2f}')
    
def weight_modification(model, quiet=False):
    rand_conn = random.choice(model.connections)

    rand_conn.weight = torch.randn(1)
    
    if not quiet:
        print(f'connection {rand_conn.innov_num} (in this nn) weight modified to {rand_conn.weight.item():.2f}')

def creates_cycle(model, source, target):
    """Returns True if adding an edge from `source` to `target` would create a cycle."""
    visited = set()

    def dfs(node):
        if node.id in visited:
            return False
        if node == source:
            return True  # Found a path back to source — would create cycle
        visited.add(node.id)
        for conn in model.connections:
            if conn.enable and conn.in_node == node:
                if dfs(conn.out_node):
                    return True
        return False

    return dfs(target)


def add_connection(model, quiet=False):
    max_attempts = 1000  # prevent infinite loop
    for _ in range(max_attempts):
        rand_node_in = random.choice(model.nodes)

        if rand_node_in.is_output:
            continue
    
        rand_node_out = random.choice(model.nodes)
        
        if rand_node_out == rand_node_in or rand_node_out.is_input:
            continue
    
        # Skip if connection already exists
        # Basically no add connection will work in the beginning because its fully connected
        if any(conn.in_node == rand_node_in and conn.out_node == rand_node_out for conn in model.connections):
            continue

        # Prevents acylic
        if model.creates_cycle(rand_node_in, rand_node_out):
            continue

        in_out_tuple = (rand_node_in.id, rand_node_out.id)

        if in_out_tuple not in NN.resulting_node_map:
            NN.resulting_node_map.update({in_out_tuple: None}) # No resulting node until it is split for the first time
            NN.innov_num_map.update({in_out_tuple: NN.next_innov_num}) # No resulting node until it is split for the first time
            NN.next_innov_num += 1

        innov_num = NN.innov_num_map[in_out_tuple]

        conn = ConnectionGene(rand_node_in, rand_node_out, innov_num, torch.randn(1))
        
        if rand_node_in.id not in model.connections_by_id:
            model.connections_by_id[rand_node_in.id ] = [conn]
        else:
            model.connections_by_id[rand_node_in.id ].append(conn)

        model.connections.append(conn)
    
        rand_node_out.num_incoming_connections += 1
        
        if not quiet:
            print(f"Connection created from node {rand_node_in.id} to node {rand_node_out.id} (innovation #{innov_num})")

        return
        
    if not quiet:
        print("Failed to add connection after max attempts.")

def add_node(model, quiet=False): 
    enabled_conn_ids = [i for i, c in enumerate(model.connections) if c.enable]
    
    if not enabled_conn_ids:
        if not quiet:
            print("No enabled connections to split.")
        return
        
    rand_conn_id = random.choice(enabled_conn_ids)

    # Splits an existing connection by adding a node
    model.connections[rand_conn_id].enable = False

    old_in_out_pair = (model.connections[rand_conn_id].in_node.id, model.connections[rand_conn_id].out_node.id)
    # Since this connection already exists, it should be in the map. Whether it is none or not is decided
    if NN.resulting_node_map[old_in_out_pair] is not None:
        new_node_id = NN.resulting_node_map[old_in_out_pair]
    else:
        new_node_id = NN.next_node_id
        NN.resulting_node_map[old_in_out_pair] = new_node_id
        NN.next_node_id += 1

    new_node = Node(new_node_id)
    model.nodes.append(new_node)

    in_out_tuple = (model.connections[rand_conn_id].in_node.id, new_node.id)

    if in_out_tuple not in NN.resulting_node_map:
        NN.resulting_node_map.update({in_out_tuple: None}) # No resulting node until it is split for the first time
        NN.innov_num_map.update({in_out_tuple: NN.next_innov_num}) # No resulting node until it is split for the first time
        NN.next_innov_num += 1

    innov_num = NN.innov_num_map[in_out_tuple]

    conn = ConnectionGene(model.connections[rand_conn_id].in_node, new_node, innov_num, torch.randn(1))
    
    if model.connections[rand_conn_id].in_node.id not in model.connections_by_id:
        model.connections_by_id[model.connections[rand_conn_id].in_node.id ] = [conn]
    else:
        model.connections_by_id[model.connections[rand_conn_id].in_node.id ].append(conn)

    model.connections.append(conn)

    in_out_tuple = (new_node.id, model.connections[rand_conn_id].out_node.id)

    if in_out_tuple not in NN.resulting_node_map:
        NN.resulting_node_map.update({in_out_tuple: None}) # No resulting node until it is split for the first time
        NN.innov_num_map.update({in_out_tuple: NN.next_innov_num}) # No resulting node until it is split for the first time
        NN.next_innov_num += 1

    innov_num = NN.innov_num_map[in_out_tuple]

    conn = ConnectionGene(new_node, model.connections[rand_conn_id].out_node, innov_num, torch.randn(1))
    
    if new_node.id not in model.connections_by_id:
        model.connections_by_id[new_node.id] = [conn]
    else:
        model.connections_by_id[new_node.id].append(conn)

    model.connections.append(conn)
    new_node.num_incoming_connections += 1

    if not quiet:
        print(f'connection {rand_conn_id} split')

def toggle_connection(model, quiet=False):
    rand_conn = random.choice(model.connections)

    rand_conn.enable = not rand_conn.enable
    
    if not rand_conn.enable:
        rand_conn.out_node.num_incoming_connections -= 1
    else:
        rand_conn.out_node.num_incoming_connections += 1

    if not quiet:
        print(f'connection {rand_conn.innov_num} (in this nn) toggled to {rand_conn.enable}')

def mutate(model, quiet=False):
    # Each one is chosen independently of each other
    # Does not include crossover. That cannot be done by itmodel

    new_model = model.clone()

    # Do mutations on new_model
    if random.random() < .8:
        weight_perturbation(new_model, quiet)
    if random.random() < .2:
        weight_modification(new_model, quiet)
    if random.random() < .1:
        add_connection(new_model, quiet)
    if random.random() < .05:
        add_node(new_model, quiet)
    if random.random() < .02:
        toggle_connection(new_model, quiet)

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