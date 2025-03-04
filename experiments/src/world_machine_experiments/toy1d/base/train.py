import os
from typing import Type

import numpy as np
import torch
from hamilton.function_modifiers import (
    datasaver, extract_fields, source, value)
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from world_machine import WorldMachine
from world_machine.train import Trainer
from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.save_train_history import (
    save_train_history)


class MSELossOnlyFirst(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, x, y):
        return self.mse(x[:, :, 0], y[:, :, 0])


@extract_fields(fields={"toy1d_model_trained": WorldMachine, "toy1d_train_history": dict[str, np.ndarray]})
def toy1d_model_training_info(toy1d_model_untrained: WorldMachine,
                              toy1d_dataloaders: dict[str, DataLoader],
                              n_epoch: int,
                              optimizer_class: Type[Optimizer],
                              learning_rate: float,
                              weight_decay: float,
                              device: str = "cpu",
                              accumulation_steps: int = 1,
                              mask_sensorial_data: float | None = None,
                              generator_numpy: np.random.Generator | None = None) -> dict[str, WorldMachine | dict[str, np.ndarray]]:

    optimizer = optimizer_class(toy1d_model_untrained.parameters(
    ), lr=learning_rate, weight_decay=weight_decay)

    toy1d_model_untrained.to(device)

    if generator_numpy is None:
        generator_numpy = np.random.default_rng(0)

    trainer = Trainer(False, mask_sensorial_data, generator_numpy)
    trainer.add_decoded_state_criterion("mse", torch.nn.MSELoss())
    trainer.add_decoded_state_criterion("mse_first", MSELossOnlyFirst(), True)

    trainer.add_sensorial_criterion("mse", "state_control", torch.nn.MSELoss())
    trainer.add_sensorial_criterion(
        "mse", "next_measurement", torch.nn.MSELoss())

    history = trainer(toy1d_model_untrained, toy1d_dataloaders,
                      optimizer, n_epoch, accumulation_steps)

    info = {"toy1d_model_trained": toy1d_model_untrained,
            "toy1d_train_history": history}

    return info


@datasaver()
def save_toy1d_model(toy1d_model_trained: WorldMachine, output_dir: str) -> dict:
    path = os.path.join(output_dir, "toy1d_model.pt")
    torch.save(toy1d_model_trained, path)

    info = {"model_path": path}

    return info


save_toy1d_train_history = function_variation({"train_history": source(
    "toy1d_train_history"), "history_name": value("toy1d_train_history")}, "save_toy1d_train_history")(save_train_history)
