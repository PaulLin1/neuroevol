import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class Innovation:
    innovation_id: int
    in_node: int
    out_node: int
    weight: float = 0.0

class NEATGenome(nn.Module):
    def __init__(self, input_size: int, output_size: int, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        
        # Pre-allocate maximum possible network size for batch processing
        self.max_nodes = 10000  # Adjust based on your needs
        self.max_connections = 50000
        
        # Connection genes as tensors for vectorized operations
        self.connection_matrix = torch.zeros(self.max_nodes, self.max_nodes, device=device)
        self.weight_matrix = torch.zeros(self.max_nodes, self.max_nodes, device=device)
        self.enabled_matrix = torch.zeros(self.max_nodes, self.max_nodes, dtype=torch.bool, device=device)
        
        # Node information
        self.node_types = torch.zeros(self.max_nodes, dtype=torch.int32, device=device)  # 0=input, 1=hidden, 2=output
        self.node_activations = torch.zeros(self.max_nodes, device=device)
        self.active_nodes = torch.zeros(self.max_nodes, dtype=torch.bool, device=device)
        
        # Topological ordering for feedforward (computed once, reused)
        self.topo_order = torch.zeros(self.max_nodes, dtype=torch.int64, device=device)
        self.topo_length = 0
        
        # Initialize basic structure
        self._initialize_minimal()
        
    def _initialize_minimal(self):
        """Initialize minimal genome with direct input->output connections"""
        # Mark input and output nodes as active
        self.active_nodes[:self.input_size] = True
        self.active_nodes[self.input_size:self.input_size + self.output_size] = True
        
        # Set node types
        self.node_types[:self.input_size] = 0  # inputs
        self.node_types[self.input_size:self.input_size + self.output_size] = 2  # outputs
        
        # Create direct connections input->output with random weights
        for i in range(self.input_size):
            for j in range(self.output_size):
                out_idx = self.input_size + j
                self.connection_matrix[i, out_idx] = 1
                self.weight_matrix[i, out_idx] = torch.randn(1, device=self.device) * 0.5
                self.enabled_matrix[i, out_idx] = True
        
        self._update_topology()
    
    def _update_topology(self):
        """Compute topological ordering for feedforward evaluation"""
        # Simple topological sort - inputs first, then hidden by layer, then outputs
        order = []
        
        # Add input nodes
        for i in range(self.input_size):
            if self.active_nodes[i]:
                order.append(i)
        
        # Add hidden nodes (simplified - assumes no cycles)
        for i in range(self.input_size, self.max_nodes):
            if self.active_nodes[i] and self.node_types[i] == 1:
                order.append(i)
        
        # Add output nodes
        for i in range(self.input_size, self.input_size + self.output_size):
            if self.active_nodes[i]:
                order.append(i)
        
        self.topo_length = len(order)
        self.topo_order[:self.topo_length] = torch.tensor(order, device=self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ultra-fast vectorized forward pass"""
        batch_size = x.shape[0]
        
        # Initialize activations for the batch
        activations = torch.zeros(batch_size, self.max_nodes, device=self.device)
        
        # Set input activations
        activations[:, :self.input_size] = x
        
        # Vectorized feedforward through topologically sorted nodes
        for i in range(self.topo_length):
            node_idx = self.topo_order[i]
            
            if self.node_types[node_idx] == 0:  # Skip input nodes
                continue
                
            # Compute weighted sum of inputs to this node
            # Only consider enabled connections
            incoming_mask = self.enabled_matrix[:, node_idx] & self.active_nodes
            
            if incoming_mask.any():
                # Vectorized computation across batch
                weighted_sum = torch.sum(
                    activations[:, incoming_mask] * 
                    self.weight_matrix[incoming_mask, node_idx].unsqueeze(0),
                    dim=1
                )
                
                # Apply activation function (tanh for speed)
                if self.node_types[node_idx] == 2:  # Output node
                    activations[:, node_idx] = weighted_sum  # Linear output
                else:  # Hidden node
                    activations[:, node_idx] = torch.tanh(weighted_sum)
        
        # Return output activations
        output_start = self.input_size
        output_end = self.input_size + self.output_size
        return activations[:, output_start:output_end]
    
    def add_node(self, connection_idx: Tuple[int, int]) -> int:
        """Add node by splitting existing connection"""
        in_node, out_node = connection_idx
        
        # Find next available node index
        new_node_idx = torch.nonzero(~self.active_nodes)[0].item()
        
        # Disable old connection
        self.enabled_matrix[in_node, out_node] = False
        
        # Add new connections
        self.connection_matrix[in_node, new_node_idx] = 1
        self.weight_matrix[in_node, new_node_idx] = 1.0  # Weight = 1
        self.enabled_matrix[in_node, new_node_idx] = True
        
        self.connection_matrix[new_node_idx, out_node] = 1
        self.weight_matrix[new_node_idx, out_node] = self.weight_matrix[in_node, out_node]
        self.enabled_matrix[new_node_idx, out_node] = True
        
        # Activate new node
        self.active_nodes[new_node_idx] = True
        self.node_types[new_node_idx] = 1  # Hidden node
        
        self._update_topology()
        return new_node_idx
    
    def add_connection(self, in_node: int, out_node: int, weight: float = None):
        """Add new connection between nodes"""
        if weight is None:
            weight = torch.randn(1, device=self.device) * 0.5
        
        self.connection_matrix[in_node, out_node] = 1
        self.weight_matrix[in_node, out_node] = weight
        self.enabled_matrix[in_node, out_node] = True
    
    def get_active_connections(self) -> torch.Tensor:
        """Return indices of active connections for mutation/crossover"""
        return torch.nonzero(self.enabled_matrix & (self.connection_matrix > 0))
    
    def clone(self) -> 'NEATGenome':
        """Fast genome cloning"""
        new_genome = NEATGenome(self.input_size, self.output_size, self.device)
        new_genome.connection_matrix = self.connection_matrix.clone()
        new_genome.weight_matrix = self.weight_matrix.clone()
        new_genome.enabled_matrix = self.enabled_matrix.clone()
        new_genome.node_types = self.node_types.clone()
        new_genome.active_nodes = self.active_nodes.clone()
        new_genome.topo_order = self.topo_order.clone()
        new_genome.topo_length = self.topo_length
        return new_genome

# Batch evaluation for population
class NEATPopulation:
    def __init__(self, genomes: List[NEATGenome]):
        self.genomes = genomes
        self.device = genomes[0].device if genomes else 'cuda'
    
    def evaluate_batch(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Evaluate entire population in parallel"""
        outputs = []
        
        # Simple parallel evaluation - can be optimized further
        with torch.no_grad():
            for genome in self.genomes:
                outputs.append(genome(x))
        
        return outputs

# Usage example for ImageNet scale:
def create_imagenet_genome(device='cuda'):
    # For ImageNet: assuming feature extraction gives 2048 features
    input_size = 2048  
    output_size = 1000  # ImageNet classes
    
    genome = NEATGenome(input_size, output_size, device)
    return genome


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np

# ===== EXAMPLE 1: Basic Usage =====
def basic_example():
    """Simple example showing basic genome operations"""
    print("=== Basic NEAT Genome Example ===")
    
    # Create a simple genome
    genome = NEATGenome(input_size=4, output_size=2, device='cuda')
    
    # Test forward pass
    batch_size = 32
    inputs = torch.randn(batch_size, 4, device='cuda')
    outputs = genome(inputs)
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    
    # Add a hidden node
    connections = genome.get_active_connections()
    if len(connections) > 0:
        conn_to_split = connections[0]  # Split first connection
        new_node = genome.add_node((conn_to_split[0].item(), conn_to_split[1].item()))
        print(f"Added hidden node: {new_node}")
    
    # Test after modification
    outputs_after = genome(inputs)
    print(f"Output shape after adding node: {outputs_after.shape}")
    
    # Clone genome
    cloned = genome.clone()
    cloned_outputs = cloned(inputs)
    print(f"Cloned genome outputs match: {torch.allclose(outputs_after, cloned_outputs)}")

# ===== EXAMPLE 2: NEAT Population Evolution =====
class SimpleNEATEvolution:
    def __init__(self, pop_size=100, input_size=4, output_size=2):
        self.pop_size = pop_size
        self.input_size = input_size
        self.output_size = output_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create initial population
        self.population = [
            NEATGenome(input_size, output_size, self.device) 
            for _ in range(pop_size)
        ]
        
        # Innovation tracking (simplified)
        self.innovation_counter = 0
    
    def evaluate_population(self, X, y):
        """Evaluate entire population on dataset"""
        fitnesses = []
        
        with torch.no_grad():
            for genome in self.population:
                outputs = genome(X)
                
                # Simple MSE fitness (lower is better, so we negate)
                mse = nn.MSELoss()(outputs, y)
                fitness = -mse.item()
                fitnesses.append(fitness)
        
        return fitnesses
    
    def select_parents(self, fitnesses, num_parents=20):
        """Select best genomes as parents"""
        # Sort by fitness (descending)
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        return [self.population[i] for i in sorted_indices[:num_parents]]
    
    def crossover(self, parent1, parent2):
        """Simple crossover between two genomes"""
        child = parent1.clone()
        
        # Random weight mixing
        mask = torch.rand_like(child.weight_matrix) > 0.5
        child.weight_matrix = torch.where(mask, parent1.weight_matrix, parent2.weight_matrix)
        
        return child
    
    def mutate(self, genome, mutation_rate=0.1):
        """Apply mutations to genome"""
        # Weight mutation
        weight_mask = torch.rand_like(genome.weight_matrix) < mutation_rate
        weight_noise = torch.randn_like(genome.weight_matrix) * 0.1
        genome.weight_matrix += weight_mask * weight_noise
        
        # Structural mutations (simplified)
        if random.random() < 0.05:  # 5% chance to add connection
            self._add_random_connection(genome)
        
        if random.random() < 0.03:  # 3% chance to add node
            self._add_random_node(genome)
    
    def _add_random_connection(self, genome):
        """Add random connection"""
        active_nodes = torch.nonzero(genome.active_nodes).flatten()
        if len(active_nodes) < 2:
            return
        
        # Pick random nodes
        in_node = random.choice(active_nodes[:genome.input_size + 10])  # Bias towards inputs
        out_node = random.choice(active_nodes[genome.input_size:])  # Only to hidden/output
        
        if not genome.enabled_matrix[in_node, out_node]:
            genome.add_connection(in_node.item(), out_node.item())
    
    def _add_random_node(self, genome):
        """Add random node by splitting connection"""
        active_connections = genome.get_active_connections()
        if len(active_connections) > 0:
            conn = random.choice(active_connections)
            genome.add_node((conn[0].item(), conn[1].item()))
    
    def evolve_generation(self, X, y):
        """Run one generation of evolution"""
        # Evaluate population
        fitnesses = self.evaluate_population(X, y)
        
        # Select parents
        parents = self.select_parents(fitnesses)
        
        # Create new population
        new_population = parents.copy()  # Keep best genomes
        
        # Fill rest with offspring
        while len(new_population) < self.pop_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
        return max(fitnesses), np.mean(fitnesses)

def evolution_example():
    """Example of NEAT evolution on XOR problem"""
    print("\n=== NEAT Evolution Example (XOR) ===")
    
    # XOR dataset
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device='cuda')
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32, device='cuda')
    
    # Create evolution engine
    neat = SimpleNEATEvolution(pop_size=50, input_size=2, output_size=1)
    
    # Evolve for several generations
    for generation in range(100):
        max_fitness, avg_fitness = neat.evolve_generation(X, y)
        
        if generation % 20 == 0:
            print(f"Generation {generation}: Max={max_fitness:.4f}, Avg={avg_fitness:.4f}")
        
        # Check if solved
        if max_fitness > -0.01:  # Very low error
            print(f"Solved XOR in generation {generation}!")
            break
    
    # Test best genome
    best_genome = neat.select_parents(neat.evaluate_population(X, y), 1)[0]
    with torch.no_grad():
        predictions = best_genome(X)
        print(f"Final predictions: {predictions.cpu().numpy().flatten()}")
        print(f"Target:           {y.cpu().numpy().flatten()}")

# ===== EXAMPLE 3: ImageNet Scale Usage =====
def imagenet_scale_example():
    """Example usage for ImageNet scale problems"""
    print("\n=== ImageNet Scale Example ===")
    
    # Simulate ImageNet features (from pre-trained CNN)
    batch_size = 256
    feature_dim = 2048  # ResNet features
    num_classes = 1000
    
    # Create genome for ImageNet
    genome = create_imagenet_genome(device='cuda')
    
    # Simulate feature extraction pipeline
    class ImageNetNEAT(nn.Module):
        def __init__(self, neat_genome):
            super().__init__()
            # Use pre-trained ResNet as feature extractor
            resnet = torchvision.models.resnet50(pretrained=True)
            self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove classifier
            self.neat_classifier = neat_genome
            
            # Freeze feature extractor
            for param in self.features.parameters():
                param.requires_grad = False
        
        def forward(self, x):
            # Extract features
            features = self.features(x)
            features = features.view(features.size(0), -1)  # Flatten
            
            # NEAT classification
            return self.neat_classifier(features)
    
    # Create full model
    model = ImageNetNEAT(genome)
    
    # Simulate batch processing
    dummy_images = torch.randn(batch_size, 3, 224, 224, device='cuda')
    
    print("Processing ImageNet-scale batch...")
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    with torch.no_grad():
        outputs = model(dummy_images)
    end_time.record()
    
    torch.cuda.synchronize()
    elapsed = start_time.elapsed_time(end_time)
    
    print(f"Batch size: {batch_size}")
    print(f"Output shape: {outputs.shape}")
    print(f"Processing time: {elapsed:.2f}ms")
    print(f"Throughput: {batch_size / (elapsed/1000):.0f} images/sec")

# ===== EXAMPLE 4: Population Batch Evaluation =====
def population_batch_example():
    """Example of evaluating entire population in batch"""
    print("\n=== Population Batch Evaluation ===")
    
    # Create population
    pop_size = 20
    population = [NEATGenome(100, 10, 'cuda') for _ in range(pop_size)]
    
    # Create batch data
    batch_size = 64
    X = torch.randn(batch_size, 100, device='cuda')
    
    # Evaluate all genomes
    print("Evaluating population...")
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    all_outputs = []
    with torch.no_grad():
        for genome in population:
            outputs = genome(X)
            all_outputs.append(outputs)
    end_time.record()
    
    torch.cuda.synchronize()
    elapsed = start_time.elapsed_time(end_time)
    
    print(f"Population size: {pop_size}")
    print(f"Batch size: {batch_size}")
    print(f"Total evaluations: {pop_size * batch_size}")
    print(f"Time: {elapsed:.2f}ms")
    print(f"Evaluations/sec: {(pop_size * batch_size) / (elapsed/1000):.0f}")

# ===== RUN ALL EXAMPLES =====
if __name__ == "__main__":
    # Make sure CUDA is available for best performance
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU (will be slower)")
    
    # Run examples
    basic_example()
    evolution_example()
    imagenet_scale_example()
    population_batch_example()
    
    print("\n=== All examples completed! ===")