import numpy as np

def measure_compatibility(g1, g2, c1, c2, c3):
    # Use dicts for fast lookup
    g1_dict = {conn.innov_num: conn.weight.item() for conn in g1.connections}
    g2_dict = {conn.innov_num: conn.weight.item() for conn in g2.connections}

    if not g1_dict and not g2_dict:
        return 0.0

    g1_keys = set(g1_dict)
    g2_keys = set(g2_dict)

    max1 = max(g1_keys, default=0)
    max2 = max(g2_keys, default=0)

    matching_keys = g1_keys & g2_keys
    if matching_keys:
        w_diff = np.fromiter((abs(g1_dict[k] - g2_dict[k]) for k in matching_keys), dtype=np.float32).mean()
    else:
        w_diff = 0.0

    # Count excess and disjoint efficiently
    excess = sum(1 for k in g1_keys if k > max2) + sum(1 for k in g2_keys if k > max1)
    disjoint = len(g1_keys ^ g2_keys) - excess

    N = max(len(g1_dict), len(g2_dict))
    N = max(N, 1) if N < 20 else N  # Avoid division by small numbers

    return (c1 * excess / N) + (c2 * disjoint / N) + (c3 * w_diff)

import random

def tournament_selection(population, tournament_size=5, num_winners=1):
    if len(population) < tournament_size:
        tournament_size = len(population)

    winners = []
    for _ in range(num_winners):
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda model_info: model_info['fitness'])
        winners.append(winner)
    return winners
