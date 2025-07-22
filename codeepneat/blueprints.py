import random
import torch.nn as nn
from codeepneat.modules import *

block_registry = {
    "conv": ConvBlock,
    "residual": ResidualBlock,
    "poolconv": PoolConvBlock,
    "mlp": MLPBlock
}

class ModuleGenome:
    def __init__(self, block_type, params=None):
        self.block_type = block_type
        self.params = params or {}

    def to_module(self):
        return block_registry[self.block_type](**self.params)

    def __repr__(self):
        return f"ModuleGenome(block_type={self.block_type}, params={self.params})"

class BlueprintGenome:
    def __init__(self, module_ids=None, max_modules=4):
        self.module_ids = module_ids or []
        self.max_modules = max_modules

    def initialize_random(self, module_pool, length_range=(2, 4)):
        length = random.randint(*length_range)
        self.module_ids = []

        non_mlp_keys = [k for k, mg in module_pool.items() if mg.block_type != "mlp"]
        mlp_keys = [k for k, mg in module_pool.items() if mg.block_type == "mlp"]

        for i in range(length):
            if i == length - 1:
                # Last module must be MLP
                if not mlp_keys:
                    raise ValueError("No MLP modules available in module_pool to assign as last module.")
                self.module_ids.append(random.choice(mlp_keys))
            else:
                # Only non-MLP modules allowed before last
                if not non_mlp_keys:
                    raise ValueError("No non-MLP modules available in module_pool for intermediate modules.")
                self.module_ids.append(random.choice(non_mlp_keys))

    def mutate(self, module_pool, mutation_rate=0.2):
        # Enforce MLP only at the end
        non_mlp_keys = [k for k, mg in module_pool.items() if mg.block_type != "mlp"]
        mlp_keys = [k for k, mg in module_pool.items() if mg.block_type == "mlp"]

        for i in range(len(self.module_ids)):
            if random.random() < mutation_rate:
                if i == len(self.module_ids) - 1:
                    # last position can be MLP or non-MLP
                    self.module_ids[i] = random.choice(list(module_pool.keys()))
                else:
                    # only non-MLP in middle
                    self.module_ids[i] = random.choice(non_mlp_keys)

        # Possibly add module at the end
        if random.random() < 0.1 and len(self.module_ids) < self.max_modules:
            # New module: MLP only allowed at the end
            # To keep consistency, append MLP or non-MLP randomly
            if random.random() < 0.3:
                # Append MLP at end
                self.module_ids.append(random.choice(mlp_keys))
            else:
                self.module_ids.append(random.choice(non_mlp_keys))

        # Possibly remove module but never remove the last if it's MLP to keep classifier logic simple
        if random.random() < 0.1 and len(self.module_ids) > 1:
            # Don't remove last MLP block if present (to keep consistent)
            if self.module_ids[-1] in mlp_keys:
                # Remove from the middle only
                idx_to_remove = random.randint(0, len(self.module_ids) - 2)
            else:
                idx_to_remove = random.randint(0, len(self.module_ids) - 1)
            self.module_ids.pop(idx_to_remove)

    def assemble_network(self, module_pool, input_shape, flatten_for_mlp=True):
        modules = []
        C, H, W = input_shape
        after_mlp = False

        for i, mid in enumerate(self.module_ids):
            mg = module_pool[mid]
            params = dict(mg.params)  # clone to avoid in-place mutation

            if after_mlp and mg.block_type != "mlp":
                raise ValueError(f"Invalid blueprint: non-MLP module after MLP at position {i}")

            if mg.block_type == "conv":
                params.setdefault("in_channels", C)
                out_channels = params.get("out_channels")
                if out_channels is None:
                    raise ValueError("conv block missing 'out_channels'")
                C = out_channels
                print(f"Module {i} (conv): C={C}, H={H}, W={W}")

            elif mg.block_type == "poolconv":
                params.setdefault("in_channels", C)
                out_channels = params.get("out_channels")
                pool_kernel = params.get("pool_kernel", 2)  # default pooling size = 2
                if out_channels is None:
                    raise ValueError("poolconv block missing 'out_channels'")
                if "pool_type" not in params:
                    raise ValueError("poolconv block missing 'pool_type'")
                C = out_channels
                H = max(1, H // pool_kernel)
                W = max(1, W // pool_kernel)
                print(f"Module {i} (poolconv): C={C}, H={H}, W={W}")

            elif mg.block_type == "residual":
                params["channels"] = C
                print(f"Module {i} (residual): C={C}, H={H}, W={W}")

            elif mg.block_type == "mlp":
                if not after_mlp:
                    input_dim = C * H * W if flatten_for_mlp else C
                    params["input_dim"] = input_dim
                    print(f"Module {i} (mlp): computed input_dim={input_dim} (C={C}, H={H}, W={W})")
                else:
                    params["input_dim"] = C
                    print(f"Module {i} (mlp): input_dim={params['input_dim']} (after previous MLP)")

                if "output_dim" not in params:
                    raise ValueError("mlp block missing 'output_dim'")

                params.setdefault("hidden_dim", max(16, (params["input_dim"] + params["output_dim"]) // 2))
                C = params["output_dim"]
                H, W = 1, 1
                after_mlp = True
                print(f"Module {i} (mlp): output_dim={params['output_dim']}")

            else:
                raise ValueError(f"Unknown block type: {mg.block_type}")

            module = ModuleGenome(mg.block_type, params).to_module()
            modules.append(module)

        class AssembledNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.mods = nn.ModuleList(modules)

            def forward(self, x):
                for i, mod in enumerate(self.mods):
                    x = mod(x)
                return x

        return AssembledNetwork()



    def __repr__(self):
        return f"BlueprintGenome(module_ids={self.module_ids})"
