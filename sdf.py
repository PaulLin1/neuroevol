import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import random
import math
import time
from neat.nn_gpu import *
from neat.mutation import *
from neat.speciation import *
random.seed(42)
torch.manual_seed(42)

# MNIST
ds = 'mnist'
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset

class Flatten:
    def __call__(self, x):
        return x.view(-1)  # flatten 1x28x28 to 784
    
transform = transforms.Compose([
    transforms.ToTensor(),    # (1, 28, 28)
    transforms.Normalize((0.1307,), (0.3081,)),  # Note: normalization expects channel dim, so do before squeeze if needed
    Flatten()
])

# Load training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# Subset stuff
train_size = len(train_dataset) // 100
test_size = len(test_dataset) // 100

train_subset = Subset(train_dataset, torch.randperm(len(train_dataset))[:train_size])
test_subset = Subset(test_dataset, torch.randperm(len(test_dataset))[:test_size])

train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(train_dataset), shuffle=False)

# Hyperparameters
population_size = 400
epochs = 200
input_dim = 28*28
output_dim = 10

tournament_size = 5 # tournament selection
top_k = 0.3 # The percentage of genomes to keep for reproduction

crossover_percent = 0.5

# hyperparameters for measuring compatibility from https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf
c1 = 1.0
c2 = 1.0
c3 = 3.0
delta_thresh = 2.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Using list of lists
# Dead species will not be kept track of. There will be no empty list
population = []

# Reset NN Class
reset_NN_class_state()

# Init first model
new_model = {"model": NN(input_dim, output_dim).to(device), "loss": float('inf'), "fitness": -float('inf')}
population.append([new_model])
with torch.no_grad():
    for _ in range(population_size - 1):
        new_model = {"model": NN(input_dim, output_dim).to(device), "loss": float('inf'), "fitness": -float('inf')}

        added = False
        for idx, species in enumerate(population):
            delta = measure_compatibility(new_model['model'], species[0]['model'], c1, c2, c3)

            if delta < delta_thresh:
                population[idx].append(new_model)
                added = True
                break
        if not added:
            # New species created
            population.append([new_model])


loss_fn = nn.CrossEntropyLoss()

print(device)

for epoch in range(epochs): 
    print('population size: ' + len(population))
    print('epoch: ' + epoch)
    time1 = time.time()


    for data_batch, label_batch in train_loader: # train_loader is one batch so i can use this hack i think idk
        data_batch = data_batch.to(device)
        label_batch = label_batch.to(device)
        
        for s in population:
            for model in s:
                output = model['model'].to(device)(data_batch)
                loss = loss_fn(output, label_batch)
                model['loss'] = loss.item()

    time2 = time.time()
    print(time2 - time1)
    # Fitness sharing
    for species in population:
        species_size = len(species)
        for genome in species:
            raw_fitness = 1 / (1 + genome['loss'])
            genome['fitness'] = raw_fitness / species_size

    f_pop = [model_info for s in population for model_info in s]
    sorted_models = sorted(f_pop, key=lambda x: x["loss"])
    # print(sorted_models[0]['loss'])
    # Last epoch do not make new models

    if epoch % 50 == 0:
        f_pop = [model_info for s in population for model_info in s]
        sorted_models = sorted(f_pop, key=lambda x: x["loss"])
        model = sorted_models[0]['model']
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                labels = labels.to(device)
                
                outputs = model(data)  # logits
                predicted = torch.argmax(outputs, dim=1)  # class indices

                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        print(f"Accuracy: {accuracy * 100:.2f}%")

    if epoch == epochs - 1:
        f_pop = [model_info for s in population for model_info in s]
        sorted_models = sorted(f_pop, key=lambda x: x["loss"])
        model = sorted_models[0]['model']
        break
    time3 = time.time()
    print(time3 - time2)

        # This is just a list not a list of lists
    new_population = []

    # Calculate total population size for normalization
    total_species_models = sum(len(species) for species in population)

    from heapq import nlargest

    for species in population:
        offspring = []

        species_size = len(species)
        offspring_count = max(1, round((species_size / total_species_models) * population_size))

        # Use heapq.nlargest for faster top-k selection (O(n log k) vs O(n log n))
        ranked_models_count = max(1, math.ceil(top_k * species_size))
        parents = nlargest(ranked_models_count, species, key=lambda x: x["fitness"])

        elite_count = max(1, int(0.1 * species_size))
        elites = nlargest(elite_count, species, key=lambda x: x["fitness"])

        # Use preallocated list + extend instead of repeated appending
        offspring.extend({
            "model": elite["model"],
            "loss": elite["loss"],
            "fitness": elite["fitness"]
        } for elite in elites)

        # Precompute how many mutated offspring we need
        to_mutate = offspring_count - len(offspring)
        if to_mutate > 0:
            sampled_parents = random.choices(parents, k=to_mutate)
            mutated_offspring = [{
                "model": mutate(parent['model'], True),
                "loss": float('inf'),
                "fitness": -float('inf')
            } for parent in sampled_parents]

            offspring.extend(mutated_offspring)

        new_population.extend(offspring)


    time4 = time.time()
    print(time4 - time3)
    # If total new_population is too big or small, adjust
    if len(new_population) > population_size:
        # Randomly truncate to population_size
        new_population = random.sample(new_population, population_size)
    elif len(new_population) < population_size:
        # Randomly duplicate some to fill
        deficit = population_size - len(new_population)
        new_population.extend(random.choices(new_population, k=deficit))

    # Now re-divide into species
    new_population_divided = []

    for model in new_population:
        if len(new_population_divided) == 0:
            new_population_divided.append([model])
        else:
            added = False
            for idx, species in enumerate(new_population_divided):
                delta = measure_compatibility(model['model'], species[0]['model'], c1, c2, c3)
                if delta < delta_thresh:
                    new_population_divided[idx].append(model)
                    added = True
                    break
            if not added:
                # New species created
                new_population_divided.append([model])
    
    time5 = time.time()
    print(time5 - time4)
    population = new_population_divided


# Evaulate


# Evaulate
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, labels in test_loader:
        data = data.to(device)
        labels = labels.to(device)
        
        outputs = model(data)  # logits
        predicted = torch.argmax(outputs, dim=1)  # class indices

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")

torch.save(model, f"models/{ds}_{population_size}_population_{epochs}_epochs_{accuracy * 100:.2f}_accuracy.pth")
