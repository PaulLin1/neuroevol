import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import random
import math
import time

from neat.cppn import *
from neat.genome import *
from neat.speciation import *

random.seed(42)
torch.manual_seed(42)

# sklearn digits

# 8x8

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()

data_tensor = torch.tensor(digits.data, dtype=torch.float32)
data_tensor = torch.tensor(digits.data / 16.0, dtype=torch.float32) # Normalize for neat
target_tensor = torch.tensor(digits.target, dtype=torch.long)

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    data_tensor, target_tensor, test_size=0.2, random_state=42, shuffle=True
)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Batch is the full size because there is no backpropogation
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))


# --------------------------------------------------------------------------------------------------------


# MNIST

# 28x28

# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from torch.utils.data import DataLoader, Subset

# transform = transforms.Compose([
#     transforms.ToTensor(),  # Converts images to PyTorch tensors
#     transforms.Normalize((0.1307,), (0.3081,))  # Mean and std dev for MNIST
# ])

# # Load training and test datasets
# train_dataset = datasets.MNIST(root='./data', train=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# # Subset stuff
# train_size = len(train_dataset) // 4
# test_size = len(test_dataset) // 4

# train_subset = Subset(train_dataset, torch.randperm(len(train_dataset))[:train_size])
# test_subset = Subset(test_dataset, torch.randperm(len(test_dataset))[:test_size])

# train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=len(train_dataset), shuffle=False)


# --------------------------------------------------------------------------------------------------------


# Init stuff

# Hyperparameters
population_size = 300
epochs = 300
input_dim = 8*8
output_dim = 10
top_k = 0.4 # The percentage of genomes to keep for reproduction
crossover_percent = 0.5

# hyperparameters for measuring compatibility from https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf
c1 = 1.0
c2 = 1.0
c3 = 3.0
delta_thresh = 3.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Using list of lists
# Dead species will not be kept track of. There will be no empty list
population = []

# Reset NN Class
reset_NN_class_state()

# Init first model
new_model = {"model": NN(input_dim, output_dim).to(device), "loss": float('inf'), "fitness": -float('inf')}
population.append([new_model])

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

# "Training" loop

for epoch in range(epochs):  
    for species in population:
        
        with torch.no_grad():
            for model_info in species:
                model_info["model"] = model_info["model"].to(device)
                model = model_info["model"]
                total_loss = 0.0
                total_samples = 0
    
                for data_batch, label_batch in train_loader:
                    data_batch = data_batch.to(device)
                    label_batch = label_batch.to(device)

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
    lowest_loss = ranked_models[0]['loss']

    # Fitness sharing
    for species in population:
        species_size = len(species)
        for genome in species:
            raw_fitness = 1 / (1 + genome['loss'])
            genome['fitness'] = raw_fitness / species_size

    # Last epoch do not make new models
    if epoch == epochs - 1:
        break

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
            offspring.append({"model": child.to(device), "loss": float('inf'), "fitness": -float('inf')})
    
        while len(offspring) != len(ranked_models):
            offspring.append({"model": random.choice(parents)['model'].mutate(True).to(device), "loss": float('inf'), "fitness": -float('inf')})
            
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
                delta = measure_compatibility(model['model'], species[0]['model'], c1, c2, c3)

                if delta < delta_thresh:
                    new_population_divided[idx].append(model)
                    added = True
                    break
            if not added:
                # New species created
                new_population_divided.append([model])
                    
    population = new_population_divided

    # To keep track of the num of species per epoch
    print(f"epoch: {epoch}")
    print(f"top model loss: {lowest_loss:.2f}")
    print(len(population))

model = ranked_models[0]['model'].to(device)

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

torch.save(model.state_dict(), f"models/sklearn_digits_300pop_300epoch_real{accuracy * 100:.2f}.pth")