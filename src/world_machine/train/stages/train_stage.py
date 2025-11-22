import abc

import numpy as np
import torch
from tensordict import TensorDict
from torch import Generator
from torch.nn import Module
from torch.optim import Optimizer

from world_machine.data import WorldMachineDataset
from world_machine.profile import profile_range
from world_machine.train.mode import DatasetPassMode
from world_machine.world_machine import WorldMachine


class TrainStage(abc.ABC):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)

        name = cls.__name__

        methods = {"pre_train": instance.pre_train,
                   "pre_batch": instance.pre_batch,
                   "pre_segment": instance.pre_segment,
                   "pre_forward": instance.pre_forward,
                   "forward": instance.forward,
                   "post_forward": instance.post_forward,
                   "post_segment": instance.post_segment,
                   "optimize": instance.optimize,
                   "post_batch": instance.post_batch,
                   "post_train": instance.post_train}

        instance.__setattr__("_with_forward", False)

        for method_name in methods:
            if method_name in cls.__dict__:
                method = methods[method_name]
                method = profile_range(
                    f"{name}_{method_name}", category="train_stage", domain="world_machine")(method)

                instance.__setattr__(method_name, method)

                if method_name == "forward":
                    instance.__setattr__("_with_forward", True)

        return instance

    def __init__(self, execution_order: float):
        super().__init__()

        self.np_generator: np.random.Generator
        self.torch_generator: Generator
        self.execution_order = execution_order

    def set_generators(self, np_generator: np.random.Generator, torch_generator: Generator):
        self.np_generator = np_generator
        self.torch_generator = torch_generator

    def pre_train(self,
                  model: WorldMachine,
                  criterions: dict[str, dict[str, Module]],
                  train_criterions: dict[str, dict[str, float]],
                  device: torch.device,
                  optimizer: Optimizer):
        ...

    def pre_batch(self, model: WorldMachine, mode: DatasetPassMode,
                  criterions: dict[str, dict[str, Module]], optimizer: Optimizer, device: torch.device, losses: dict, train_criterions: dict[str, dict[str, float]]) -> None:
        ...

    def pre_segment(self, itens: list[TensorDict], losses: dict, batch_size: int,
                    seq_len: int, epoch_index: int, device: torch.device,
                    state_size: int, mode: DatasetPassMode, model: WorldMachine) -> None:
        ...

    def pre_forward(self, item_index: int,  itens: list[TensorDict], mode: DatasetPassMode, batch_size: int, device: torch.device, epoch_index: int) -> None:
        ...

    def forward(self, model: WorldMachine, segment: TensorDict,  mode: DatasetPassMode) -> None:
        ...

    def post_forward(self, item_index: int,  itens: list[TensorDict], dataset: WorldMachineDataset, losses: dict, mode: DatasetPassMode) -> None:
        ...

    def post_segment(self, itens: list[TensorDict], losses: dict, dataset: WorldMachineDataset,
                     epoch_index: int, criterions: dict[str, dict[str, Module]], mode: DatasetPassMode,
                     device: torch.device, train_criterions: dict[str, dict[str, float]]) -> None:
        ...

    def optimize(self, model: WorldMachine, optimizer: Optimizer, batch_index: int, n_batch: int, losses: dict, mode: DatasetPassMode) -> None:
        ...

    def post_batch(self,
                   model: WorldMachine,
                   losses: dict,
                   criterions: dict[str, dict[str, Module]],
                   train_criterions: dict[str, dict[str, float]],
                   mode: DatasetPassMode) -> None:
        ...

    def post_train(self,
                   model: WorldMachine,
                   criterions: dict[str, dict[str, Module]],
                   train_criterions: dict[str, dict[str, float]],
                   optimizer: Optimizer):
        ...
