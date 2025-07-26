import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Load digits dataset
digits = load_digits()
X = digits.images.reshape(-1, 64).astype('float32') / 16.0  # normalize to [0, 1]
y = digits.target

# Convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create TensorDatasets (already in memory)
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)


# In-memory DataLoaders (no on-the-fly transformation needed)
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(train_dataset))

import torch
import torch.nn.functional as F
import random
from neat.genome import Genome, InnovationTracker
import torch.nn as nn

# Simple fitness function: accuracy on a small validation batch
def evaluate_fitness(genome, data_loader, device, quiet=False):
    genome.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")  # sum to average later
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = genome(X_batch)

            loss = loss_fn(outputs, y_batch)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / total if total > 0 else float('inf')
    accuracy = correct / total if total > 0 else 0.0

    # In NEAT, fitness is often defined so higher is better, so negative loss is convenient:
    fitness = -avg_loss

    # Optionally print or log metrics:
    if not quiet:
        print(f"Eval - Accuracy: {accuracy:.4f}, Avg Loss: {avg_loss:.4f}, Fitness: {fitness:.4f}")

    return fitness 

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

# Simple NEAT evolutionary loop
def neat_evolution_loop(train_loader, val_loader,
                        population_size, generations, device,
                        selection_frac=0, mutation_probs=None,
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
    population = [Genome(8*8, 10, device=device, innovation_tracker=innovation_tracker)
                  for _ in range(population_size)]

    best_genome = None
    best_fitness = -float('inf')

    for gen in range(generations):
        print(f"\nGeneration {gen+1}")

        # Evaluate fitness
        fitnesses = []
 
        for i, genome in enumerate(population):
            fitness = evaluate_fitness(genome, train_loader, device, quiet)
            fitnesses.append((fitness, genome))
            # if not quiet:
            #     print(f"  Genome {i}: fitness={fitness:.3f}")
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
    
        print(f"Best fitness: {best_fitness:.3f}")

    final_accuracy = full_evaluation(best_genome, val_loader, device)
    return best_genome, final_accuracy


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# best_genome, final_accuracy = neat_evolution_loop(train_loader, val_loader, 100, 10, device, quiet=True)
# print("Best model achieved accuracy:", print(final_accuracy))

import cProfile
cProfile.run("neat_evolution_loop(train_loader, val_loader, 500, 400, device, quiet=True)", sort="time")
