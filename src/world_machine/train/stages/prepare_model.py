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

    def pre_batch(self, model: WorldMachine, mode: DatasetPassMode,
                  criterions: dict[str, dict[str, Module]], optimizer: Optimizer, device: torch.device, losses: dict, train_criterions: dict[str, dict[str, float]]) -> None:

        self.original_grad_state = torch.is_grad_enabled()
        self.original_model_state = model.training

        if mode == DatasetPassMode.MODE_EVALUATE:
            model.eval()
            torch.set_grad_enabled(False)
        elif mode == DatasetPassMode.MODE_TRAIN:
            model.train()
            torch.set_grad_enabled(True)
            optimizer.zero_grad()
        else:
            raise ValueError(f"Unknown mode: {mode}.")

    def post_batch(self,
                   model: WorldMachine,
                   losses: dict,
                   criterions: dict[str, dict[str, Module]],
                   train_criterions: dict[str, dict[str, float]],
                   mode: DatasetPassMode) -> None:
        # Return original state
        torch.set_grad_enabled(self.original_grad_state)

        if self.original_model_state:
            model.train()
        else:
            model.eval()
