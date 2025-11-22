import abc

import numpy as np
import torch
from tensordict import TensorDict
from torch import Generator
from torch.nn import Module
from torch.optim import Optimizer

from world_machine.data import WorldMachineDataset
from world_machine.train.mode import DatasetPassMode
from world_machine.world_machine import WorldMachine

from .simple_optimizer import SimpleOptimizer


class GradientAccumulator(SimpleOptimizer):
    def __init__(self, accumulation_steps: int):
        super().__init__()

        self.accumulation_steps = accumulation_steps

    def optimize(self, model: WorldMachine, optimizer: Optimizer, batch_index: int, n_batch: int, losses: dict, mode: DatasetPassMode) -> None:
        if mode == DatasetPassMode.MODE_TRAIN:
            optimizer_loss: torch.Tensor = losses["optimizer_loss"]
            optimizer_loss_step = optimizer_loss / self.accumulation_steps
            optimizer_loss_step.backward()

            if (((batch_index+1) % self.accumulation_steps == 0) or
                    (batch_index+1 == n_batch)):

                optimizer.step()
                optimizer.zero_grad()
