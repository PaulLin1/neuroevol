import time
import torch

# MNIST
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
import torch

# Transform MNIST
transform = transforms.Compose([
    transforms.ToTensor(),  # Shape becomes (1, 8, 8), values in [0, 1]
    transforms.Lambda(lambda x: x.view(-1))  # Flatten to shape (64,)
])

# Load datasets with transform
train_data = MNIST(root="./data", train=True, download=True, transform=transform)
val_data = MNIST(root="./data", train=False, download=True, transform=transform)

# Dataloaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)

import torch
import torch.nn.functional as F
import random
from genome import Genome, InnovationTracker

# Simple fitness function: accuracy on a small validation batch
def evaluate_fitness(genome, data_loader, device):
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
            break  # Only evaluate on 1 batch for speed
    return correct / total if total > 0 else 0.0

# Simple NEAT evolutionary loop
def neat_evolution_loop(train_loader, val_loader,
                        population_size, generations, device,
                        selection_frac=0.4, mutation_probs=None,
                        use_crossover=False, quiet=False):
    if mutation_probs is None:
        mutation_probs = {
            'weight_mutate': 0.8,
            'weight_reset': 0.1,
            'add_connection': 0.05,
            'add_node': 0.02
        }

    innovation_tracker = InnovationTracker()

    # Initialize population
    population = [Genome(28*28, 10, device=device, innovation_tracker=innovation_tracker)
                  for _ in range(population_size)]

    best_genome = None
    best_fitness = -float('inf')

    for gen in range(generations):
        if not quiet:
            print(f"\nGeneration {gen+1}")

        # Evaluate fitness
        fitnesses = []
        for i, genome in enumerate(population):
            fitness = evaluate_fitness(genome, val_loader, device)
            fitnesses.append((fitness, genome))
            if not quiet:
                print(f"  Genome {i}: fitness={fitness:.3f}")
            if fitness > best_fitness:
                best_fitness = fitness
                best_genome = genome

        # Sort and select top
        fitnesses.sort(key=lambda x: x[0], reverse=True)
        num_selected = max(2, int(selection_frac * population_size))
        selected = [g for _, g in fitnesses[:num_selected]]

        if not quiet:
            print(f"  Selected top {num_selected} genomes")

        # Create new population
        new_population = [selected[0]]  # Elitism

        while len(new_population) < population_size:
            if use_crossover:
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                child = parent1.crossover(parent2)
            else:
                parent = random.choice(selected)
                child = parent.clone()

            # Mutations
            if random.random() < mutation_probs['weight_mutate']:
                child.mutate_weights(perturb_chance=0.9, reset_chance=0.1)
            if random.random() < mutation_probs['add_connection']:
                child.mutate_add_connection()
            if random.random() < mutation_probs['add_node']:
                child.mutate_add_node()

            new_population.append(child)

        population = new_population

    if not quiet:
        print("Evolution complete.")
        print(f"Best fitness: {best_fitness:.3f}")

    print(best_fitness)
    return best_genome, best_fitness


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
# best_genome, best_fitness = neat_evolution_loop(train_loader, val_loader, device)
# print("Best model achieved fitness:", best_fitness)

import cProfile
cProfile.run("neat_evolution_loop(train_loader, val_loader, 100, 500, device, quiet=True)", sort="time")