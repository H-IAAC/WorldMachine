import torch
from torch.optim import Optimizer

from world_machine.profile import profile_range
from world_machine.train.mode import DatasetPassMode
from world_machine.world_machine import WorldMachine

from .train_stage import TrainStage


class SimpleOptimizer(TrainStage):
    def __init__(self):
        super().__init__(5)

    def optimize(self, model: WorldMachine, optimizer: Optimizer, batch_index: int, n_batch: int, losses: dict, mode: DatasetPassMode) -> None:
        if mode == DatasetPassMode.MODE_TRAIN:
            optimizer_loss: torch.Tensor = losses["optimizer_loss"]

            with profile_range("SimpleOptimizer_backward", category="train_stage", domain="world_machine"):
                optimizer_loss.backward()

            with profile_range("SimpleOptimizer_optim_step", category="train_stage", domain="world_machine"):
                optimizer.step()

            with profile_range("SimpleOptimizer_optim_zero_grad", category="train_stage", domain="world_machine"):
                optimizer.zero_grad()
