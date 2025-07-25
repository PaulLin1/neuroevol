import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class InnovationTracker:
    def __init__(self):
        self.current_innovation = 0
        self.innovations = {}  # (src, dst) -> innovation number

    def get_innovation(self, src, dst):
        key = (int(src), int(dst))
        if key not in self.innovations:
            self.current_innovation += 1
            self.innovations[key] = self.current_innovation
        return self.innovations[key]

class Genome(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, device='cuda', innovation_tracker=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.innovation_tracker = innovation_tracker or InnovationTracker()

        # Nodes
        self.input_nodes = torch.arange(input_dim, device=device)
        self.output_nodes = torch.arange(input_dim, input_dim + output_dim, device=device)
        self.node_count = input_dim + output_dim
        self.all_nodes = list(range(self.node_count))

        # Edges - fully connected input->output
        src = self.input_nodes.repeat_interleave(output_dim)
        dst = self.output_nodes.repeat(input_dim)
        self.edge_index = torch.stack([src, dst], dim=0)  # [2, E]
        num_edges = self.edge_index.size(1)

        # Edge attributes: weights as a Parameter tensor
        self.edge_weight = nn.Parameter(torch.randn(num_edges, device=device) * 0.01)

        # Edge enabled mask as buffer (non-trainable)
        self.register_buffer('edge_enabled', torch.ones(num_edges, dtype=torch.bool, device=device))

        # Innovation numbers as buffer for indexing & comparison
        innovations = torch.tensor([self.innovation_tracker.get_innovation(s.item(), d.item())
                                    for s, d in zip(src, dst)],
                                   dtype=torch.long, device=device)
        self.register_buffer('edge_innovations', innovations)

        # Bias not part of vanilla neat but nice
        self.bias = nn.Parameter(torch.zeros(self.node_count, device=device))

    def build_adjacency_matrix(self):
        # Only enabled edges
        mask = self.edge_enabled
        src = self.edge_index[0, mask]
        dst = self.edge_index[1, mask]
        weights = self.edge_weight[mask]
        
        indices = torch.stack([dst, src], dim=0)  # transpose src->dst to dst->src for multiplication
        values = weights
        size = (self.node_count, self.node_count)
        A = torch.sparse_coo_tensor(indices, values, size, device=self.device)
        return A
    
    def forward(self, x, steps=3):
        batch_size = x.size(0)
        activations = torch.zeros(batch_size, self.node_count, device=self.device, dtype=x.dtype)
        activations[:, self.input_nodes] = x

        A = self.build_adjacency_matrix()

        for _ in range(steps):
            # Sparse matmul: (N_nodes x N_nodes) @ (B x N_nodes)^T -> (N_nodes x B), then transpose back
            activations = torch.sparse.mm(A, activations.T).T + self.bias
            activations.relu_()
            activations[:, self.input_nodes] = x  # clamp inputs

        return activations[:, self.output_nodes]

    def mutate_weights(self, perturb_chance=0.8, perturb_std=0.1, reset_chance=0.1):
        with torch.no_grad():
            rand_vals = torch.rand(self.edge_weight.size(), device=self.device)
            # Perturb weights where rand in [reset_chance, perturb_chance)
            perturb_mask = (rand_vals < perturb_chance) & (rand_vals >= reset_chance)
            # Reset weights where rand < reset_chance
            reset_mask = rand_vals < reset_chance

            noise = torch.randn_like(self.edge_weight) * perturb_std
            self.edge_weight[perturb_mask] += noise[perturb_mask]
            self.edge_weight[reset_mask] = torch.randn(reset_mask.sum(), device=self.device) * 0.1

    def mutate_add_connection(self, max_tries=100):
        existing = set(zip(self.edge_index[0].tolist(), self.edge_index[1].tolist()))
        tries = 0

        dst_candidates = [n for n in self.all_nodes if n not in self.input_nodes]
        if not dst_candidates:
            return False  # no valid destination nodes

        while tries < max_tries:
            tries += 1
            src = random.choice(self.all_nodes)
            dst = random.choice(dst_candidates)
            if src == dst or (src, dst) in existing:
                continue
            innov = self.innovation_tracker.get_innovation(src, dst)
            self._add_edge(src, dst, innov, weight=torch.randn(1, device=self.device) * 0.1)
            return True
        return False

    def mutate_add_node(self):
        enabled_indices = torch.nonzero(self.edge_enabled).squeeze(-1).tolist()
        if not enabled_indices:
            return False

        edge_idx = random.choice(enabled_indices)
        src = int(self.edge_index[0, edge_idx].item())
        dst = int(self.edge_index[1, edge_idx].item())

        # Disable old edge
        self.edge_enabled[edge_idx] = False

        # Add new node id
        new_node = self.node_count
        self.node_count += 1
        self.all_nodes.append(new_node)

        # Extend bias with zeros (detach to avoid breaking graph)
        new_bias = torch.zeros(1, device=self.device)
        self.bias = nn.Parameter(torch.cat([self.bias.detach(), new_bias], dim=0))

        # Add edges src->new_node and new_node->dst
        innov1 = self.innovation_tracker.get_innovation(src, new_node)
        innov2 = self.innovation_tracker.get_innovation(new_node, dst)

        self._add_edge(src, new_node, innov1, weight=torch.tensor([1.0], device=self.device))
        self._add_edge(new_node, dst, innov2, weight=self.edge_weight[edge_idx].detach().clone())

        return True

    def _add_edge(self, src, dst, innovation_num, weight=None):
        with torch.no_grad():
            new_edge = torch.tensor([[src], [dst]], device=self.device)
            self.edge_index = torch.cat([self.edge_index, new_edge], dim=1)

            if weight is None:
                weight = torch.randn(1, device=self.device) * 0.1
            else:
                weight = weight.reshape(-1)

            self.edge_weight = nn.Parameter(torch.cat([self.edge_weight.detach(), weight], dim=0))

            self.edge_enabled = torch.cat([self.edge_enabled, torch.tensor([True], device=self.device)])

            new_innov = torch.tensor([innovation_num], device=self.device, dtype=torch.long)
            self.edge_innovations = torch.cat([self.edge_innovations, new_innov])

    def crossover(self, other):
        # Helper: build dict of innovation -> (src, dst, weight, enabled)
        def build_edge_dict(genome):
            return {
                int(innov): (
                    int(genome.edge_index[0, i]),
                    int(genome.edge_index[1, i]),
                    genome.edge_weight[i].detach().item(),
                    bool(genome.edge_enabled[i].item())
                )
                for i, innov in enumerate(genome.edge_innovations.tolist())
            }

        self_edges = build_edge_dict(self)
        other_edges = build_edge_dict(other)

        all_innovs = set(self_edges.keys()) | set(other_edges.keys())

        child = Genome(self.input_dim, self.output_dim, device=self.device, innovation_tracker=self.innovation_tracker)
        # Reset child's edges to empty
        child.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        child.edge_weight = nn.Parameter(torch.empty((0,), device=self.device))
        child.edge_enabled = torch.empty((0,), dtype=torch.bool, device=self.device)
        child.edge_innovations = torch.empty((0,), dtype=torch.long, device=self.device)

        child.all_nodes = list(set(self.all_nodes) | set(other.all_nodes))
        child.node_count = max(child.all_nodes) + 1
        child.bias = nn.Parameter(torch.zeros(child.node_count, device=self.device))

        for innov in sorted(all_innovs):
            if innov in self_edges and innov in other_edges:
                chosen = self_edges[innov] if random.random() < 0.5 else other_edges[innov]
                # Enabled if both parents enabled; otherwise 25% chance enabled
                enabled = self_edges[innov][3] and other_edges[innov][3]
                if not enabled and random.random() < 0.75:
                    enabled = False
            elif innov in self_edges:
                chosen = self_edges[innov]
                enabled = chosen[3]
            else:
                # Discard excess/exclusive edges from other parent
                continue

            s, d, w, _ = chosen
            new_edge = torch.tensor([[s], [d]], device=self.device)
            child.edge_index = torch.cat([child.edge_index, new_edge], dim=1)
            child.edge_weight = nn.Parameter(torch.cat([child.edge_weight.detach(), torch.tensor([w], device=self.device)], dim=0))
            child.edge_enabled = torch.cat([child.edge_enabled, torch.tensor([enabled], device=self.device)])
            child.edge_innovations = torch.cat([child.edge_innovations, torch.tensor([innov], device=self.device)])

        return child

    def distance(self, other, c1=1.0, c2=1.0, c3=0.4):
        self_innovs = set(self.edge_innovations.tolist())
        other_innovs = set(other.edge_innovations.tolist())

        max_innov_self = max(self_innovs) if self_innovs else 0
        max_innov_other = max(other_innovs) if other_innovs else 0
        max_innov = max(max_innov_self, max_innov_other)

        # Matching innovations
        matching_innovs = self_innovs.intersection(other_innovs)

        # Disjoint = in one genome but not the other, excluding excess
        disjoint_innovs = (self_innovs.symmetric_difference(other_innovs)) - \
                          {i for i in range(max_innov + 1) if i not in self_innovs and i not in other_innovs}

        # Excess = innovation numbers above max innovation of other genome
        excess_innovs = {i for i in (self_innovs.union(other_innovs))
                         if i > max_innov_self or i > max_innov_other}

        def get_weight(genome, innov):
            idx = (genome.edge_innovations == innov).nonzero(as_tuple=True)[0].item()
            return genome.edge_weight[idx].item()

        weight_diffs = [abs(get_weight(self, innov) - get_weight(other, innov)) for innov in matching_innovs]
        avg_weight_diff = sum(weight_diffs) / len(weight_diffs) if weight_diffs else 0.0

        N = max(len(self_innovs), len(other_innovs))
        if N < 20:
            N = 1

        dist = (c1 * len(excess_innovs) / N) + (c2 * len(disjoint_innovs) / N) + (c3 * avg_weight_diff)
        return dist

    def clone(self):
        # Create new instance without calling __init__
        new = object.__new__(Genome)
        nn.Module.__init__(new)  # initialize Module internals

        # Core attributes copy
        new.device = self.device
        new.input_dim = self.input_dim
        new.output_dim = self.output_dim
        new.innovation_tracker = self.innovation_tracker

        # Node info (copy lists)
        new.input_nodes = self.input_nodes.clone()
        new.output_nodes = self.output_nodes.clone()
        new.node_count = self.node_count
        new.all_nodes = self.all_nodes[:]

        # Edge info
        new.edge_index = self.edge_index.clone()
        new.edge_weight = nn.Parameter(self.edge_weight.detach().clone())
        # Buffers must be registered
        new.register_buffer('edge_enabled', self.edge_enabled.clone())
        new.register_buffer('edge_innovations', self.edge_innovations.clone())

        # Bias
        new.bias = nn.Parameter(self.bias.detach().clone())

        return new
