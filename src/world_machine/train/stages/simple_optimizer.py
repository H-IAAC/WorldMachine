import abc

import numpy as np
import torch
from tensordict import TensorDict
from torch import Generator
from torch.nn import Module
from torch.optim import Optimizer

from world_machine import WorldMachine
from world_machine.data import WorldMachineDataset
from world_machine.train import DatasetPassMode

from .train_stage import TrainStage


class SimpleOptimizer(TrainStage):
    def __init__(self):
        super().__init__(5)

    def optimize(self, model: WorldMachine, optimizer: Optimizer, batch_index: int, n_batch: int, losses: dict, mode: DatasetPassMode) -> None:
        if mode == DatasetPassMode.MODE_TRAIN:
            optimizer_loss: torch.Tensor = losses["optimizer_loss"]
            optimizer_loss.backward()

            optimizer.step()
            optimizer.zero_grad()
