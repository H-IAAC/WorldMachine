import torch
from tensordict import TensorDict
from torch.nn import Module
from torch.optim import Optimizer

from world_machine import WorldMachine
from world_machine.data import WorldMachineDataset
from world_machine.train import DatasetPassMode

from .train_stage import TrainStage


class StateManager(TrainStage):
    def __init__(self, stable_state_epochs: int = 1):
        super().__init__(1)

        self._stable_state_epochs = stable_state_epochs

    def pre_segment(self, itens: list[TensorDict], losses: dict, batch_size: int,
                    seq_len: int, epoch_index: int, device: torch.device, state_size: int, mode: DatasetPassMode) -> None:

        if epoch_index == 0:
            for item in itens:
                seq_len = item["inputs"][next(
                    iter(item["inputs"].keys()))].shape[1]

                state = torch.rand(
                    (batch_size, seq_len, state_size), device=device, generator=self.torch_generator)
                state = (2*state)-1

                item["inputs"]["state"] = state

    def post_segment(self, itens: list[TensorDict], losses: dict, dataset: WorldMachineDataset, epoch_index: int,
                     criterions: dict[str, dict[str, Module]], mode: DatasetPassMode,
                     device: torch.device, train_criterions: dict[str, dict[str, float]]) -> None:

        for item in itens:
            state = item["inputs"]["state"]
            state_next = item["logits"]["state"]

            state_current = torch.roll(state_next, 1, 1)
            state_current[:, 0] = state[:, 0]

            indexes = item["index"]

            if (epoch_index % self._stable_state_epochs == 0):
                dataset.set_state(indexes, state_current)
