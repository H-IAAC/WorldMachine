import os
from typing import Type

import numpy as np
import pysdtw
import torch
from hamilton.function_modifiers import (
    datasaver, extract_fields, source, value)
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from world_machine import WorldMachine
from world_machine.train import CriterionSet, ParameterScheduler, Trainer
from world_machine.train.stages import (
    GradientAccumulator, LocalSetter, LossManager, NoiseAdder, SensorialMasker,
    SequenceBreaker, ShortTimeRecaller, StateManager, StateSaveMethod)
from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.save_metrics import save_metrics
from world_machine_experiments.toy1d.dimensions import Dimensions


class MSELossOnlyFirst(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, x, y):
        return self.mse(x[:, :, 0], y[:, :, 0])


class MeanSoftDTW(torch.nn.Module):
    def __init__(self, gamma=.1, scale: float = 1, use_cuda=False):
        super().__init__()

        self._use_cuda = use_cuda

        self.stdw = pysdtw.SoftDTW(gamma=.1, use_cuda=use_cuda)

        self.scale = scale

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self._use_cuda:
            x = x.cpu()
            y = y.cpu()
        elif x.device.type == "cpu" or y.device.type == "cpu":
            raise ValueError(
                "Inputs must be in CUDA device when 'use_cuda' = True")

        input1 = torch.cat([x, x, y])
        input2 = torch.cat([y, x, y])

        loss: torch.Tensor = self.stdw(input1, input2)

        loss_xy, loss_xx, loss_yy = torch.split(loss, x.shape[0])

        loss = loss_xy - 1/2*(loss_xx+loss_yy)

        loss = loss.mean()
        loss *= self.scale
        loss = loss.to(x.device)

        return loss


def toy1d_criterion_set(sensorial_train_losses: set[Dimensions] = set(), train_mse: bool = True, train_sdtw: bool = False) -> CriterionSet:
    cs = CriterionSet()

    cs.add_decoded_state_criterion("mse", torch.nn.MSELoss(), train_mse)
    # cs.add_decoded_state_criterion(
    #    "0.1sdtw", MeanSoftDTW(scale=.1, use_cuda=True), train_sdtw)

    # cs.add_sensorial_criterion("mse", "state_control", torch.nn.MSELoss(
    # ), train=(Dimensions.STATE_CONTROL in sensorial_train_losses))
    cs.add_sensorial_criterion(
        "mse", "measurement", torch.nn.MSELoss(), train=(Dimensions.MEASUREMENT in sensorial_train_losses and train_mse))
    # cs.add_sensorial_criterion("0.1sdtw", "measurement", MeanSoftDTW(scale=.1, use_cuda=True), train=(
    #    Dimensions.MEASUREMENT in sensorial_train_losses and train_sdtw))

    return cs


@extract_fields(fields={"toy1d_model_trained": WorldMachine, "toy1d_train_history": dict[str, np.ndarray], "toy1d_trainer": Trainer})
def toy1d_model_training_info(toy1d_model_untrained: WorldMachine,
                              toy1d_dataloaders: dict[str, DataLoader],
                              toy1d_criterion_set: CriterionSet,
                              n_epoch: int,
                              optimizer_class: Type[Optimizer],
                              learning_rate: float,
                              weight_decay: float,
                              device: str = "cpu",
                              accumulation_steps: int = 1,
                              mask_sensorial_data:  float | None | dict[str, float |
                                                                        ParameterScheduler] | ParameterScheduler = None,
                              discover_state: bool = False,
                              state_save_method: StateSaveMethod = StateSaveMethod.REPLACE,
                              stable_state_epochs: int = 1,
                              seed: int | list[int] = 0,
                              n_segment: int = 1,
                              fast_forward: bool = False,
                              short_time_recall: set[Dimensions] = set(),
                              recall_n_past: int = 2,
                              recall_n_future: int = 2,
                              recall_stride_past: int = 1,
                              recall_stride_future: int = 1,
                              measurement_size: int = 2,
                              state_regularizer: str | None = None,
                              check_input_masks: bool = False,
                              state_cov_regularizer: float | None = None,
                              state_dimensions: list[int] | None = None,
                              noise_config: dict[str,
                                                 dict[str, float]] | None = None,
                              local_chance: float | None = None) -> dict[str, WorldMachine | dict[str, np.ndarray] | Trainer]:

    optimizer = optimizer_class(toy1d_model_untrained.parameters(
    ), lr=learning_rate, weight_decay=weight_decay)

    toy1d_model_untrained.to(device)

    decoded_state_size = len(
        state_dimensions) if state_dimensions is not None else 3

    stages = []

    if accumulation_steps != 1:
        stages.append(GradientAccumulator(accumulation_steps))
    if mask_sensorial_data != None:
        stages.append(SensorialMasker(mask_sensorial_data))
    if discover_state:
        stages.append(StateManager(stable_state_epochs,
                      check_input_masks, state_save_method))
    if n_segment != 1:
        stages.append(SequenceBreaker(n_segment, fast_forward))
    if len(short_time_recall) != 0:
        dimension_sizes = {}
        criterions = {}

        for dim in short_time_recall:
            if dim == Dimensions.STATE_DECODED:
                dimension_sizes["state_decoded"] = decoded_state_size
                criterions["state_decoded"] = torch.nn.MSELoss()
            elif dim == Dimensions.MEASUREMENT:
                dimension_sizes["measurement"] = measurement_size
                criterions["measurement"] = torch.nn.MSELoss()

        stages.append(ShortTimeRecaller(dimension_sizes=dimension_sizes,
                                        criterions=criterions,
                                        n_past=recall_n_past,
                                        n_future=recall_n_future,
                                        stride_past=recall_stride_past,
                                        stride_future=recall_stride_future))

    if noise_config is not None:
        means = {}
        stds = {}
        mins = {}
        maxs = {}

        for name in noise_config:
            means[name] = noise_config[name]["mean"]
            stds[name] = noise_config[name]["std"]

            mins[name] = -1
            maxs[name] = 1

        stages.append(NoiseAdder(means, stds, mins, maxs))

    if local_chance is not None:
        print("With Local")
        stages.append(LocalSetter(local_chance))

    stages.append(LossManager(state_regularizer,
                  state_cov_regularizer, multiply_target_masks=False))

    trainer = Trainer(toy1d_criterion_set, stages, seed)

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


save_toy1d_train_history = function_variation({"metrics": source(
    "toy1d_train_history"), "metrics_name": value("toy1d_train_history")}, "save_toy1d_train_history")(save_metrics)
