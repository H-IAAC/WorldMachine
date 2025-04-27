import os
from typing import Type

import numpy as np
import torch
from hamilton.function_modifiers import (
    datasaver, extract_fields, source, value)
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from world_machine import WorldMachine
from world_machine.train import ParameterScheduler, Trainer
from world_machine.train.stages import (
    GradientAccumulator, LossManager, SensorialMasker, SequenceBreaker,
    ShortTimeRecaller, StateManager)
from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.save_train_history import (
    save_train_history)
from world_machine_experiments.toy1d.dimensions import Dimensions


class MSELossOnlyFirst(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, x, y):
        return self.mse(x[:, :, 0], y[:, :, 0])


@extract_fields(fields={"toy1d_model_trained": WorldMachine, "toy1d_train_history": dict[str, np.ndarray], "toy1d_trainer": Trainer})
def toy1d_model_training_info(toy1d_model_untrained: WorldMachine,
                              toy1d_dataloaders: dict[str, DataLoader],
                              n_epoch: int,
                              optimizer_class: Type[Optimizer],
                              learning_rate: float,
                              weight_decay: float,
                              device: str = "cpu",
                              accumulation_steps: int = 1,
                              mask_sensorial_data:  float | None | dict[str, float |
                                                                        ParameterScheduler] | ParameterScheduler = None,
                              discover_state: bool = False,
                              stable_state_epochs: int = 1,
                              sensorial_train_losses: set[Dimensions] = set(),
                              seed: int | list[int] = 0,
                              n_segment: int = 1,
                              fast_forward: bool = False,
                              short_time_recall: set[Dimensions] = set(),
                              recall_n_past: int = 2,
                              measurement_size: int = 2,
                              state_regularizer: str | None = None,
                              check_input_masks: bool = False,
                              state_cov_regularizer: float | None = None) -> dict[str, WorldMachine | dict[str, np.ndarray] | Trainer]:

    optimizer = optimizer_class(toy1d_model_untrained.parameters(
    ), lr=learning_rate, weight_decay=weight_decay)

    toy1d_model_untrained.to(device)

    stages = []

    if accumulation_steps != 1:
        stages.append(GradientAccumulator(accumulation_steps))
    if mask_sensorial_data != None:
        stages.append(SensorialMasker(mask_sensorial_data))
    if discover_state:
        stages.append(StateManager(stable_state_epochs, check_input_masks))
    if n_segment != 1:
        stages.append(SequenceBreaker(n_segment, fast_forward))
    if len(short_time_recall) != 0:
        dimension_sizes = {}
        criterions = {}

        for dim in short_time_recall:
            if dim == Dimensions.STATE_DECODED:
                dimension_sizes["state_decoded"] = 3
                criterions["state_decoded"] = MSELossOnlyFirst()
            elif dim == Dimensions.NEXT_MEASUREMENT:
                dimension_sizes["next_measurement"] = measurement_size
                criterions["next_measurement"] = MSELossOnlyFirst()

        stages.append(ShortTimeRecaller(recall_n_past, dimension_sizes,
                                        criterions=criterions))

    stages.append(LossManager(state_regularizer, state_cov_regularizer))

    trainer = Trainer(stages, seed)
    trainer.add_decoded_state_criterion("mse", torch.nn.MSELoss())
    trainer.add_decoded_state_criterion("mse_first", MSELossOnlyFirst(), True)

    trainer.add_sensorial_criterion("mse", "state_control", torch.nn.MSELoss(
    ), train=(Dimensions.STATE_CONTROL in sensorial_train_losses))
    trainer.add_sensorial_criterion(
        "mse", "next_measurement", MSELossOnlyFirst(), train=(Dimensions.NEXT_MEASUREMENT in sensorial_train_losses))

    history = trainer(toy1d_model_untrained, toy1d_dataloaders,
                      optimizer, n_epoch)

    info = {"toy1d_model_trained": toy1d_model_untrained,
            "toy1d_train_history": history,
            "toy1d_trainer": trainer}

    return info


@datasaver()
def save_toy1d_model(toy1d_model_trained: WorldMachine, output_dir: str) -> dict:
    path = os.path.join(output_dir, "toy1d_model.pt")
    torch.save(toy1d_model_trained, path)

    info = {"model_path": path}

    return info


save_toy1d_train_history = function_variation({"train_history": source(
    "toy1d_train_history"), "history_name": value("toy1d_train_history")}, "save_toy1d_train_history")(save_train_history)
