from collections import defaultdict

import torch
from tensordict import TensorDict
from torch.nn import Module
from torch.optim import Optimizer

from world_machine.data import WorldMachineDataset
from world_machine.train.mode import DatasetPassMode
from world_machine.world_machine import WorldMachine

from .train_stage import TrainStage


class NoiseAdder(TrainStage):
    def __init__(self, means: dict[str, float], stds: dict[str, float], mins: dict[str, float], maxs: dict[str, float]):
        super().__init__(0.5)

        self.means = means
        self.stds = stds
        self.mins = defaultdict(None, mins)
        self.maxs = defaultdict(None, maxs)

    def pre_segment(self, itens: list[TensorDict], losses: dict, batch_size: int,
                    seq_len: int, epoch_index: int, device: torch.device, state_size: int, mode: DatasetPassMode, model: WorldMachine) -> None:

        if mode == DatasetPassMode.MODE_TRAIN:
            item = itens[0]

            with torch.no_grad():
                for key in self.means:
                    if key == "state" and epoch_index == 0 and key not in item["inputs"]:
                        continue

                    noise = torch.empty_like(item["inputs"][key]).normal_(
                        self.means[key], self.stds[key], generator=self.torch_generator)

                    item["inputs"][key] += noise

                    if key in self.mins or key in self.maxs:
                        item["inputs"][key].clamp(
                            self.mins[key], self.maxs[key])
