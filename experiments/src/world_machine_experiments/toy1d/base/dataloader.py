import torch
from torch.utils.data import DataLoader

from world_machine.data import WorldMachineDataLoader

from .dataset import Toy1dDataset


def toy1d_dataloaders(toy1d_datasets: dict[str, Toy1dDataset],
                      batch_size: int,
                      generator_torch: torch.Generator | None) -> dict[str, DataLoader]:

    dataloaders = {}
    for name in toy1d_datasets:
        dataloaders[name] = WorldMachineDataLoader(toy1d_datasets[name],
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   num_workers=0,
                                                   generator=generator_torch)

    return dataloaders
