import torch

from world_machine.data import WorldMachineDataLoader, WorldMachineDataset


class BenchmarkDataset(WorldMachineDataset):
    def __init__(self):
        sensorial_dimensions = ["dim0", "dim1"]
        size = 32*10
        has_state_decoded = False
        has_masks = True
        super().__init__(sensorial_dimensions, size, has_state_decoded, has_masks)

    def get_dimension_item(self, dimension: str, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty([100, 3]), torch.empty([100, 3])

    def get_dimension_mask(self, dimension, index) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones(100, dtype=bool), torch.ones(100, dtype=bool)


def get_benchmark_dataloaders():
    dataset = BenchmarkDataset()

    train_loader = WorldMachineDataLoader(dataset, 32, True)
    val_loader = WorldMachineDataLoader(dataset, 32, True)

    return train_loader, val_loader
