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


class TrainStage(abc.ABC):

    def __init__(self, execution_order: float):
        super().__init__()

        self.np_generator: np.random.Generator
        self.torch_generator: Generator
        self.execution_order = execution_order

    def set_generators(self, np_generator: np.random.Generator, torch_generator: Generator):
        self.np_generator = np_generator
        self.torch_generator = torch_generator

    def pre_batch(self, model: WorldMachine, mode: DatasetPassMode,
                  criterions: dict[str, dict[str, Module]], optimizer: Optimizer, device: torch.device, losses: dict) -> None:
        ...

    def pre_segment(self, itens: list[TensorDict], losses: dict, batch_size: int,
                    seq_len: int, epoch_index: int, device: torch.device,
                    state_size: int, mode: DatasetPassMode) -> None:
        ...

    def pre_forward(self, item_index: int,  itens: list[TensorDict], mode: DatasetPassMode, batch_size: int, device: torch.device, epoch_index: int) -> None:
        ...

    def post_forward(self, item_index: int,  itens: list[TensorDict], dataset: WorldMachineDataset, losses: dict, mode: DatasetPassMode) -> None:
        ...

    def post_segment(self, itens: list[TensorDict], losses: dict, dataset: WorldMachineDataset,
                     epoch_index: int, criterions: dict[str, dict[str, Module]], mode: DatasetPassMode,
                     device: torch.device, train_criterions: dict[str, dict[str, float]]) -> None:
        ...

    def optimize(self, model: WorldMachine, optimizer: Optimizer, batch_index: int, n_batch: int, losses: dict, mode: DatasetPassMode) -> None:
        ...

    def post_batch(self, model: WorldMachine, losses: dict) -> None:
        ...
