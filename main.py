import time
import torch

# MNIST
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
import torch

# # Transform MNIST
# transform = transforms.Compose([
#     transforms.ToTensor(),  # Shape becomes (1, 8, 8), values in [0, 1]
#     transforms.Lambda(lambda x: x.view(-1))  # Flatten to shape (64,)
# ])

# # Load datasets with transform
# train_data = MNIST(root="./data", train=True, download=True, transform=transform)
# val_data = MNIST(root="./data", train=False, download=True, transform=transform)

# # Dataloaders
# train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=64)


def load_data_to_memory(dataset):
    Xs = []
    ys = []
    for x, y in dataset:
        Xs.append(x)
        ys.append(y)
    Xs = torch.stack(Xs)
    ys = torch.tensor(ys)
    return Xs, ys

# Load MNIST without transform
train_data_raw = MNIST(root="./data", train=True, download=True, transform=None)
val_data_raw = MNIST(root="./data", train=False, download=True, transform=None)

# Preprocess once: convert to tensors and flatten
to_tensor_flatten = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

def preprocess_dataset(raw_dataset):
    Xs, ys = [], []
    for img, label in raw_dataset:
        tensor_img = to_tensor_flatten(img)
        Xs.append(tensor_img)
        ys.append(label)
    return torch.stack(Xs), torch.tensor(ys)

train_X, train_y = preprocess_dataset(train_data_raw)
val_X, val_y = preprocess_dataset(val_data_raw)

# Create TensorDataset for easy batching
train_dataset = TensorDataset(train_X, train_y)
val_dataset = TensorDataset(val_X, val_y)

# DataLoaders that just batch from tensors already in RAM
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


import torch
import torch.nn.functional as F
import random
from neat.genome import Genome, InnovationTracker
# import torch.nn as nn

# # Simple fitness function: accuracy on a small validation batch
# def evaluate_fitness(genome, data_loader, device, quiet=False):
#     genome.eval()
#     loss_fn = nn.CrossEntropyLoss(reduction="sum")  # sum to average later
#     total_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for X_batch, y_batch in data_loader:
#             X_batch = X_batch.to(device)
#             y_batch = y_batch.to(device)
#             outputs = genome(X_batch)

#             loss = loss_fn(outputs, y_batch)
#             total_loss += loss.item()

#             preds = outputs.argmax(dim=1)
#             correct += (preds == y_batch).sum().item()
#             total += y_batch.size(0)

#     avg_loss = total_loss / total if total > 0 else float('inf')
#     accuracy = correct / total if total > 0 else 0.0

#     # In NEAT, fitness is often defined so higher is better, so negative loss is convenient:
#     fitness = -avg_loss

#     # Optionally print or log metrics:
#     if not quiet:
#         print(f"Eval - Accuracy: {accuracy:.4f}, Avg Loss: {avg_loss:.4f}, Fitness: {fitness:.4f}")

#     return fitness 

# def full_evaluation(genome, data_loader, device):
#     genome.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for X_batch, y_batch in data_loader:
#             X_batch = X_batch.to(device)
#             y_batch = y_batch.to(device)
#             outputs = genome(X_batch)
#             preds = outputs.argmax(dim=1)
#             correct += (preds == y_batch).sum().item()
#             total += y_batch.size(0)
#     return correct / total if total > 0 else 0.0

# # Simple NEAT evolutionary loop
# def neat_evolution_loop(train_loader, val_loader,
#                         population_size, generations, device,
#                         selection_frac=0, mutation_probs=None,
#                         use_crossover=False, quiet=False):
#     if mutation_probs is None:
#         mutation_probs = {
#             'weight_mutate': 0.8,
#             'weight_reset': 0.2,
#             'add_connection': 0.1,
#             'add_node': 0.05
#         }

#     innovation_tracker = InnovationTracker()

#     # Initialize population
#     population = [Genome(28*28, 10, device=device, innovation_tracker=innovation_tracker)
#                   for _ in range(population_size)]

#     best_genome = None
#     best_fitness = -float('inf')

#     for gen in range(generations):
#         print(f"\nGeneration {gen+1}")

#         # Evaluate fitness
#         fitnesses = []
 
#         for i, genome in enumerate(population):
#             fitness = evaluate_fitness(genome, train_loader, device, quiet)
#             fitnesses.append((fitness, genome))
#             # if not quiet:
#             #     print(f"  Genome {i}: fitness={fitness:.3f}")
#             if fitness > best_fitness:
#                 best_fitness = fitness
#                 best_genome = genome

#         # Sort and select top
#         fitnesses.sort(key=lambda x: x[0], reverse=True)
#         num_selected = max(2, int(selection_frac * population_size))
#         selected = [g for _, g in fitnesses[:num_selected]]

#         if not quiet:
#             print(f"  Selected top {num_selected} genomes")

#         # Create new population
#         new_population = [selected[0]]  # Elitism

#         while len(new_population) < population_size:
#             if use_crossover:
#                 parent1 = random.choice(selected)
#                 parent2 = random.choice(selected)
#                 child = parent1.crossover(parent2)
#             else:
#                 parent = random.choice(selected)
#                 child = parent.clone()

#             # Mutations
#             if random.random() < mutation_probs['weight_mutate']:
#                 child.mutate_weights(perturb_chance=0.9, reset_chance=0.1)
#             if random.random() < mutation_probs['add_connection']:
#                 child.mutate_add_connection()
#             if random.random() < mutation_probs['add_node']:
#                 child.mutate_add_node()

#             new_population.append(child)
        
#         population = new_population
    
#         print(f"Best fitness: {best_fitness:.3f}")

#     final_accuracy = full_evaluation(best_genome, val_loader, device)
#     return best_genome, final_accuracy

import copy
from concurrent.futures import ProcessPoolExecutor
def evaluate_genome_fitness(data):
    genome, X_batch, y_batch = data
    import torch
    import torch.nn as nn

    device = torch.device("cpu")  # Force CPU to avoid CUDA issues across processes
    genome = genome.to(device)
    genome.eval()

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        outputs = genome(X_batch.to(device))
        loss = loss_fn(outputs, y_batch.to(device))
        preds = outputs.argmax(dim=1)
        correct = (preds == y_batch).sum().item()
        total = y_batch.size(0)

    avg_loss = loss.item() / total
    return -avg_loss  # NEAT-style: higher fitness = better

# Full evaluation (unchanged)
def full_evaluation(genome, data_loader, device):
    genome.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = genome(X_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total if total > 0 else 0.0

# Parallel genome evaluation
def evaluate_population_parallel(population, X_batch, y_batch, max_workers=8):
    inputs = [
        (copy.deepcopy(genome).cpu(), X_batch.cpu(), y_batch.cpu())
        for genome in population
    ]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        fitnesses = list(executor.map(evaluate_genome_fitness, inputs))
    return fitnesses

# Main NEAT loop with parallel eval
def neat_evolution_loop(train_loader, val_loader,
                        population_size, generations, device,
                        selection_frac=0.2, mutation_probs=None,
                        use_crossover=False, quiet=False):

    if mutation_probs is None:
        mutation_probs = {
            'weight_mutate': 0.8,
            'weight_reset': 0.2,
            'add_connection': 0.1,
            'add_node': 0.05
        }

    innovation_tracker = InnovationTracker()

    # Initialize population
    population = [Genome(28*28, 10, device=device, innovation_tracker=innovation_tracker)
                  for _ in range(population_size)]

    best_genome = None
    best_fitness = -float('inf')

    for gen in range(generations):
        print(f"\nGeneration {gen+1}")

        # Get a fixed batch once per generation
        X_batch, y_batch = next(iter(train_loader))

        # Parallel evaluation
        fitness_values = evaluate_population_parallel(population, X_batch, y_batch)

        # Track fitness and best
        fitnesses = []
        for i, (fitness, genome) in enumerate(zip(fitness_values, population)):
            fitnesses.append((fitness, genome))
            if fitness > best_fitness:
                best_fitness = fitness
                best_genome = genome

        # Sort and select
        fitnesses.sort(key=lambda x: x[0], reverse=True)
        num_selected = max(2, int(selection_frac * population_size))
        selected = [g for _, g in fitnesses[:num_selected]]

        if not quiet:
            print(f"  Selected top {num_selected} genomes")

        # Next generation
        new_population = [selected[0]]  # Elitism

        while len(new_population) < population_size:
            if use_crossover:
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                child = parent1.crossover(parent2)
            else:
                parent = random.choice(selected)
                child = parent.clone()

            if random.random() < mutation_probs['weight_mutate']:
                child.mutate_weights(perturb_chance=0.9, reset_chance=mutation_probs['weight_reset'])
            if random.random() < mutation_probs['add_connection']:
                child.mutate_add_connection()
            if random.random() < mutation_probs['add_node']:
                child.mutate_add_node()

            new_population.append(child)

        population = new_population
        print(f"  Best fitness so far: {best_fitness:.4f}")

    final_accuracy = full_evaluation(best_genome, val_loader, device)
    return best_genome, final_accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# best_genome, final_accuracy = neat_evolution_loop(train_loader, val_loader, 10, 100, device, quiet=True)
# print("Best model achieved accuracy:", print(final_accuracy))

import cProfile
cProfile.run("neat_evolution_loop(train_loader, val_loader, 3, 3, device)", sort="time")
