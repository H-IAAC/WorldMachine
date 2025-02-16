import numpy as np
import torch


def generator_numpy(seed:int|list[int]) -> np.random.Generator:
    return np.random.default_rng(seed=seed)

def _seed_to_torch(seed:int|list[int]) -> int:
    if isinstance(seed, int):
        return seed
    
    return int(np.random.default_rng(seed).random(1)[0]*1e10)

def generator_torch(seed:int|list[int]) -> torch.Generator:
    seed = _seed_to_torch(seed)
    
    g = torch.Generator()
    g.manual_seed(seed)

    return g

def h_ensure_random_seed(seed:int|list[int]) -> None:
    seed = _seed_to_torch(seed)

    torch.manual_seed(seed)