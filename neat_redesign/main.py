import time
import torch

# # SKLEARN
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from torch.utils.data import TensorDataset, DataLoader
# import numpy as np

# # Load sklearn digits data
# digits = load_digits()
# X = digits.images  # shape (1797, 8, 8)
# y = digits.target  # shape (1797,)

# # Normalize pixels to [0,1] float
# X = X.astype(np.float32) / 16.0  # original max pixel is 16

# # Flatten: (1797, 8, 8) → (1797, 64)
# X = X.reshape(len(X), -1)

# # Convert to torch tensors
# X_tensor = torch.from_numpy(X)
# y_tensor = torch.from_numpy(y).long()

# # Train/val split
# X_train, X_val, y_train, y_val = train_test_split(
#     X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y
# )

# # Create datasets and loaders
# train_dataset = TensorDataset(X_train, y_train)
# val_dataset = TensorDataset(X_val, y_val)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64)

# MNIST
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
import torch

# Transform MNIST to match sklearn digits
transform = transforms.Compose([
    # transforms.Resize((8, 8), interpolation=InterpolationMode.BILINEAR),  # Resize 28x28 → 8x8
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

# Assume digits dataset and DataLoader as you provided are already defined

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
def neat_evolution_loop(train_loader, val_loader, device,
                        population_size=10, generations=5,
                        selection_frac=0.4, mutation_probs=None):
    if mutation_probs is None:
        mutation_probs = {
            'weight_mutate': 0.8,
            'weight_reset': 0.1,
            'add_connection': 0.05,
            'add_node': 0.02
        }

    innovation_tracker = InnovationTracker()

    # Initialize population
    population = [Genome(28*28, 10, device=device, innovation_tracker=innovation_tracker) for _ in range(population_size)]

    best_genome = None
    best_fitness = -float('inf')

    for gen in range(generations):
        print(f"\nGeneration {gen+1}")

        # Evaluate fitness for each genome
        fitnesses = []
        for i, genome in enumerate(population):
            fitness = evaluate_fitness(genome, val_loader, device)
            fitnesses.append((fitness, genome))
            print(f"  Genome {i}: fitness={fitness:.3f}")

            # Track global best
            if fitness > best_fitness:
                best_fitness = fitness
                best_genome = genome

        # Sort by fitness descending
        fitnesses.sort(key=lambda x: x[0], reverse=True)

        # Select top performers
        num_selected = max(2, int(selection_frac * population_size))
        selected = [g for _, g in fitnesses[:num_selected]]

        print(f"  Selected top {num_selected} genomes")

        # Reproduce new population
        new_population = []

        # Keep best genome unchanged (elitism)
        new_population.append(selected[0])

        while len(new_population) < population_size:
            # Crossover two random parents from selected
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)

            # Perform crossover (parent1 assumed fitter or equal)
            child = parent1.crossover(parent2)

            # Mutations on child
            if random.random() < mutation_probs['weight_mutate']:
                child.mutate_weights(perturb_chance=0.9, reset_chance=0.1)
            if random.random() < mutation_probs['add_connection']:
                child.mutate_add_connection()
            if random.random() < mutation_probs['add_node']:
                child.mutate_add_node()

            new_population.append(child)

        population = new_population

    print("Evolution complete.")
    print(f"Best fitness: {best_fitness:.3f}")
    return best_genome, best_fitness


# Run the simple NEAT evolutionary loop
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
best_genome, best_fitness = neat_evolution_loop(train_loader, val_loader, device)
print("Best model achieved fitness:", best_fitness)