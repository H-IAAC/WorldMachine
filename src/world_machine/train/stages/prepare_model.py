import torch
from torch.nn import Module
from torch.optim import Optimizer

from world_machine.train.mode import DatasetPassMode
from world_machine.world_machine import WorldMachine

from .train_stage import TrainStage


class PrepareModel(TrainStage):
    def __init__(self):
        super().__init__(0)

        self.original_model_state: bool
        self.original_grad_state: bool

    def pre_batch(self, model: WorldMachine,
                  mode: DatasetPassMode,
                  criterions: dict[str, dict[str, Module]] | None,
                  optimizer: Optimizer | None,
                  device: torch.device | None,
                  losses: dict | None,
                  train_criterions: dict[str, dict[str, float]] | None) -> None:

        self.original_grad_state = torch.is_grad_enabled()
        self.original_model_state = model.training

        if mode == DatasetPassMode.MODE_EVALUATE:
            model.eval()
            torch.set_grad_enabled(False)
        elif mode == DatasetPassMode.MODE_TRAIN:
            model.train()
            torch.set_grad_enabled(True)
            if optimizer is not None:
                optimizer.zero_grad()

        else:
            raise ValueError(f"Unknown mode: {mode}.")

    def post_batch(self,
                   model: WorldMachine,
                   losses: dict | None,
                   criterions: dict[str, dict[str, Module]] | None,
                   train_criterions: dict[str, dict[str, float]] | None,
                   mode: DatasetPassMode | None) -> None:
        # Return original state
        torch.set_grad_enabled(self.original_grad_state)

        if self.original_model_state:
            model.train()
        else:
            model.eval()
