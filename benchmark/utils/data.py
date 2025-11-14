import torch

from world_machine.data import WorldMachineDataLoader, WorldMachineDataset


class BenchmarkDataset(WorldMachineDataset):
    def __init__(self, n_batch: int = 10):
        sensory_channels = ["channel0", "channel1"]
        size = 32*n_batch
        has_state_decoded = False
        has_masks = True
        super().__init__(sensory_channels, size, has_state_decoded, has_masks)

    def get_channel_item(self, channel: str, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty([100, 3]), torch.empty([100, 3])

    def get_channel_mask(self, channel, index) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones(100, dtype=bool), torch.ones(100, dtype=bool)


def get_benchmark_dataloaders(n_batch: int = 10):
    dataset = BenchmarkDataset(n_batch)

    train_loader = WorldMachineDataLoader(dataset, 32, True)
    val_loader = WorldMachineDataLoader(dataset, 32, True)

    return train_loader, val_loader
