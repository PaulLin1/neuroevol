import random
import torch
import torch.nn as nn

from codeepneat.blueprints import *

def genome_distance(bp1, bp2):
    mods1 = bp1.module_ids
    mods2 = bp2.module_ids
    max_len = max(len(mods1), len(mods2))
    diff = sum(m1 != m2 for m1, m2 in zip(mods1, mods2))
    diff += abs(len(mods1) - len(mods2))
    return diff / max_len if max_len > 0 else 0

def speciate(population, compatibility_threshold=0.5):
    species = []
    for indiv in population:
        found = False
        for s in species:
            if genome_distance(indiv, s[0]) < compatibility_threshold:
                s.append(indiv)
                found = True
                break
        if not found:
            species.append([indiv])
    return species

def crossover(bp1, bp2):
    length = min(len(bp1.module_ids), len(bp2.module_ids))
    if length < 2:
        child_modules = bp1.module_ids.copy()
    else:
        cx_point = random.randint(1, length - 1)
        child_modules = bp1.module_ids[:cx_point] + bp2.module_ids[cx_point:]
    return BlueprintGenome(module_ids=child_modules)

def evaluate_fitness(bp, module_pool, train_loader, val_loader, device, epochs=3):
    model = bp.assemble_network(module_pool, input_shape=(1, 28, 28)).to(device)

    # Optional: Compile the model (PyTorch 2.0+ only)
    model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    fitness = correct / total
    return fitness, model

def run_evolution(
    module_pool,
    train_loader,
    val_loader,
    population_size=20,
    generations=10,
    mutation_rate=0.3,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    # Initialize population randomly
    population = []
    for _ in range(population_size):
        bp = BlueprintGenome()
        bp.initialize_random(module_pool)
        population.append(bp)

    best_model = None
    best_blueprint = None
    best_fitness = -1

    for gen in range(generations):
        print(f"Generation {gen}")

        fitness_models = []
        for i, indiv in enumerate(population):
            fitness, trained_model = evaluate_fitness(indiv, module_pool, train_loader, val_loader, device)
            fitness_models.append((fitness, trained_model, indiv))
            print(f"  Individual {i}: fitness={fitness:.4f}")

        # Sort by fitness descending
        fitness_models.sort(key=lambda x: x[0], reverse=True)
        population = [bp for _, _, bp in fitness_models]

        # Keep track of best model overall
        if fitness_models[0][0] > best_fitness:
            best_fitness = fitness_models[0][0]
            best_model = fitness_models[0][1]
            best_blueprint = fitness_models[0][2]

        # Speciate for info only (optional)
        species = speciate(population)
        print(f"  Number of species: {len(species)}")

        # Selection: top 50%
        survivors = population[: population_size // 2]

        # Create offspring by crossover + mutation until population full
        offspring = []
        while len(survivors) + len(offspring) < population_size:
            parents = random.sample(survivors, 2)
            child = crossover(parents[0], parents[1])
            child.mutate(module_pool, mutation_rate=mutation_rate)
            offspring.append(child)

        population = survivors + offspring

    print(f"Best fitness after {generations} generations: {best_fitness:.4f}")
    return best_blueprint, best_model
