def measure_compatibility(genome1, genome2, c1, c2, c3):
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
        
    return delta