import torch
from torch.utils.data import DataLoader

from .dataset import Toy1dDataset

def toy1d_dataloaders(toy1d_datasets:dict[str, Toy1dDataset],
                      batch_size:int) -> dict[str, DataLoader]:
    dataloaders = {}
    for name in toy1d_datasets:
        dataloaders[name] = DataLoader(toy1d_datasets[name], 
                                    batch_size=batch_size, 
                                    collate_fn=lambda x: torch.stack([sample for sample in x]), 
                                    shuffle=True, 
                                    drop_last=True,
                                    num_workers=0)
    
    return dataloaders