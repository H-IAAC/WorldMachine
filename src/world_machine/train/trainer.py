import time
from typing import Callable

import numba
import numpy as np
import torch
import tqdm
from numpy.typing import ArrayLike
from tensordict import TensorDict
from torch.nn import Module
from torch.utils.data import DataLoader

from world_machine.profile import profile_range
from world_machine.train.criterion_set import CriterionSet
from world_machine.train.stages import (
    LossManager, PrepareModel, SimpleOptimizer, TrainStage,
    WorldMachineForward)
from world_machine.world_machine import WorldMachine

from .mode import DatasetPassMode
from .scheduler import ParameterScheduler

try:
    import wandb
except ImportError:
    wanbd = None

# Pure Torch is more slow, cannot make work without iteration per batch


class Trainer:
    def __init__(self, criterion_set: CriterionSet, stages: list[TrainStage] | None = None, seed: list[int] | int = 0):

        self._generator_numpy = np.random.default_rng(seed=seed)
        self._generator_torch = torch.Generator()
        if isinstance(seed, list):
            seed = int(self._generator_numpy.random(1)[0]*1e10)

        self.torch_seed = seed
        self._generator_torch.manual_seed(seed)

        self._criterion_set = criterion_set

        if stages is None:
            stages = []

        with_loss_manager = False
        with_prepare_model = False
        with_optimizer = False
        with_forward = False
        for stage in stages:
            if isinstance(stage, LossManager):
                with_loss_manager = True
            if isinstance(stage, PrepareModel):
                with_prepare_model = True
            if isinstance(stage, SimpleOptimizer):
                with_optimizer = True
            if stage._with_forward:
                with_forward = True

        if not with_loss_manager:
            stages.append(LossManager())
        if not with_prepare_model:
            stages.append(PrepareModel())
        if not with_optimizer:
            stages.append(SimpleOptimizer())
        if not with_forward:
            stages.append(WorldMachineForward())

        for stage in stages:
            stage.np_generator = self._generator_numpy
            stage.torch_generator = self._generator_torch

        self._stages = stages

    def add_decoded_state_criterion(self, name: str,
                                    criterion: Module,
                                    train: bool = False,
                                    weight: float = 1.0) -> None:
        self._criterion_set.criterions["state_decoded"][name] = criterion

        if train:
            self._criterion_set.train_criterions["state_decoded"][name] = weight

    def add_sensory_criterion(self, name: str, sensory_channel: str, criterion: Module, train: bool = False,
                              weight: float = 1.0) -> None:
        if sensory_channel not in self._criterion_set.criterions:
            self._criterion_set.criterions[sensory_channel] = {}
        if sensory_channel not in self._criterion_set.train_criterions:
            self._criterion_set.train_criterions[sensory_channel] = {}

        self._criterion_set.criterions[sensory_channel][name] = criterion

        if train:
            self._criterion_set.train_criterions[sensory_channel][name] = weight

    def __call__(self, wm: WorldMachine,
                 dataloaders: dict[str, DataLoader],
                 optimizer: torch.optim.Optimizer,
                 n_epoch: int) -> dict[str, np.ndarray | dict[str, np.ndarray] | dict[str, dict[str, np.ndarray]]]:

        return self._train(wm, dataloaders, optimizer, n_epoch, False, None)

    def _compute_loss_and_optimize(self,
                                   model: WorldMachine,
                                   loader: DataLoader,
                                   mode: int = DatasetPassMode.MODE_EVALUATE,
                                   optimizer: torch.optim.Optimizer | None = None) -> dict[str, torch.Tensor]:
        """
        Computes the loss from a model across a dataset.

        If in train mode also runs optimizer steps.

        Args:
            model (torch.nn.Module): model to evaluate.
            loader (DataLoader): dataset.
            mode (int): mode of the computation. 
                        If DatasetPassMode.MODE_EVALUATE, computes without gradient, in eval mode and detachs loss.
                        If DatasetPassMode.MODE_TRAIN, computes with gradient and in train mode.
                        Default is DatasetPassMode.MODE_EVALUATE.
            optimizer (torch.optim.Optimizer, optional): optimizer to use in the train mode.

        Returns:
            torch.Tensor: resulting loss.
        """
        range_name = "train"
        if mode == DatasetPassMode.MODE_EVALUATE:
            range_name = "evaluate"

        with profile_range(range_name, category="trainer", domain="world_machine"):

            device = next(iter(model.parameters())).device
            losses = {}

            if device != self._generator_torch.device:
                self._generator_torch = torch.Generator(device=device)
                self._generator_torch.manual_seed(self.torch_seed)
                for stage in self._stages:
                    stage.torch_generator = self._generator_torch

            batch_index = 0

            with profile_range("pre_batch", category="trainer", domain="world_machine"):
                for stage in self._stages:
                    stage.pre_batch(model, mode, self._criterion_set.criterions,
                                    optimizer, device, losses, self._criterion_set.train_criterions)

            n_batch = len(loader)

            for item in tqdm.tqdm(loader):
                item = item.to(device)

                itens = [item]
                batch_size = item["inputs"].batch_size[0]
                seq_len = item["inputs"][next(
                    iter(item["inputs"].keys()))].shape[1]
                epoch_index = self._epoch_index
                state_size = model._state_size
                dataset = loader.dataset

                with profile_range("pre_segment", category="trainer", domain="world_machine"):
                    for stage in self._stages:
                        stage.pre_segment(itens, losses, batch_size,
                                          seq_len, epoch_index, device, state_size, mode, model)

                for segment_index, segment in enumerate(itens):

                    with profile_range("pre_forward", category="trainer", domain="world_machine"):
                        for stage in self._stages:
                            stage.pre_forward(
                                segment_index, itens, mode, batch_size, device, epoch_index)

                    # Forward
                    with profile_range("forward", category="trainer", domain="world_machine"):
                        for stage in self._stages:
                            stage.forward(model, segment, mode)

                    with profile_range("post_forward", category="trainer", domain="world_machine"):
                        for stage in reversed(self._stages):
                            stage.post_forward(segment_index, itens,
                                               dataset, losses, mode)

                with profile_range("post_segment", category="trainer", domain="world_machine"):
                    for stage in reversed(self._stages):
                        stage.post_segment(itens, losses, dataset,
                                           epoch_index, self._criterion_set.criterions, mode, device, self._criterion_set.train_criterions)

                with profile_range("optimize", category="trainer", domain="world_machine"):
                    for stage in reversed(self._stages):
                        stage.optimize(model, optimizer, batch_index,
                                       n_batch, losses, mode)

            with profile_range("post_batch", category="trainer", domain="world_machine"):
                for stage in reversed(self._stages):
                    stage.post_batch(model, losses, self._criterion_set.criterions,
                                     self._criterion_set.train_criterions, mode)

        return losses

    def _train(self, wm: WorldMachine,
               dataloaders: dict[str, DataLoader],
               optimizer: torch.optim.Optimizer,
               n_epoch: int,
               use_wandb: bool = False,
               early_stop: Callable[[float], bool] | None = None) -> dict[str, np.ndarray]:

        device = next(iter(wm.parameters())).device

        self._stages.sort(key=lambda s: s.execution_order)

        with profile_range("pre_train", category="trainer", domain="world_machine"):
            for stage in self._stages:
                stage.pre_train(wm, self._criterion_set.criterions,
                                self._criterion_set.train_criterions, device, optimizer)

        self._epoch_index = 0

        hist: dict[str, np.ndarray | dict[str, np.ndarray]
                   | dict[str, dict[str, np.ndarray]]] = {}
        for channel in self._criterion_set.criterions:
            hist[channel] = {}
            for criterion_name in self._criterion_set.criterions[channel]:
                hist[channel][criterion_name] = {"train": np.empty(n_epoch),
                                                 "val": np.empty(n_epoch)}

        hist["optimizer_loss"] = {"train": np.empty(n_epoch),
                                  "val": np.empty(n_epoch)}

        hist["duration"] = np.empty(n_epoch)

        loss_val = self._compute_loss_and_optimize(
            wm, dataloaders["val"], DatasetPassMode.MODE_EVALUATE)

        print("VAL ", end="")
        print_info(loss_val["optimizer_loss"], -1, n_epoch)
        for epoch in range(n_epoch):
            start_time = time.time()

            loss_train = self._compute_loss_and_optimize(
                wm, dataloaders["train"], DatasetPassMode.MODE_TRAIN, optimizer)

            end_time = time.time()

            epoch_duration = end_time - start_time

            print_info(loss_train["optimizer_loss"],
                       epoch, n_epoch, epoch_duration)

            # Validation stats
            loss_val = self._compute_loss_and_optimize(
                wm, dataloaders["val"], DatasetPassMode.MODE_EVALUATE)

            print("VAL ", end="")
            print_info(loss_val["optimizer_loss"], epoch, n_epoch)

            # Save history and log
            log: dict[str, float] = {}

            for channel in self._criterion_set.criterions:
                for criterion_name in self._criterion_set.criterions[channel]:
                    hist[channel][criterion_name]["train"][epoch] = loss_train[f"{channel}_{criterion_name}"].item(
                    )
                    hist[channel][criterion_name]["val"][epoch] = loss_val[f"{channel}_{criterion_name}"].item(
                    )

                    log[f"loss_train_{channel}_{criterion_name}"] = loss_train[f"{channel}_{criterion_name}"].item(
                    )
                    log[f"loss_val_{channel}_{criterion_name}"] = loss_val[f"{channel}_{criterion_name}"].item(
                    )

            hist["optimizer_loss"]["train"][epoch] = loss_train["optimizer_loss"].item()
            hist["optimizer_loss"]["val"][epoch] = loss_val["optimizer_loss"].item()

            log["loss_train_optimizer_loss"] = loss_train["optimizer_loss"].item()
            log["loss_val_optimizer_loss"] = loss_val["optimizer_loss"].item()

            hist["duration"][epoch] = epoch_duration

            if use_wandb:
                wandb.log(log)

            if (early_stop is not None and
                    early_stop(loss_val["optimizer_loss"])):
                break

            self._epoch_index += 1

        with profile_range("format_history", category="trainer", domain="world_machine"):
            result = {}
            result["optimizer_loss_train"] = hist["optimizer_loss"]["train"]
            result["optimizer_loss_val"] = hist["optimizer_loss"]["val"]
            result["duration"] = hist["duration"]
            for channel in self._criterion_set.criterions:
                for criterion_name in self._criterion_set.criterions[channel]:
                    result[f"{channel}_{criterion_name}_train"] = hist[channel][criterion_name]["train"]
                    result[f"{channel}_{criterion_name}_val"] = hist[channel][criterion_name]["val"]

        with profile_range("post_train", category="trainer", domain="world_machine"):
            for stage in reversed(self._stages):
                stage.post_train(wm, self._criterion_set.criterions,
                                 self._criterion_set.train_criterions, optimizer)

        return result


def print_info(loss_value: torch.Tensor, epoch: int, total_epochs: int,
               time: float | None = None):
    """
    Prints the information of a epoch.

    Args:
        loss_value (torch.Tensor): epoch loss.
        epoch (int): epoch number.
        total_epochs (int): total number of epochs. 
        time (float, optional): time to run the epoch. Don't print if is 0.0. Defaults to 0.0.
        accuracy (float, optional): epoch accuracy.
    """

    print(f'Epoch [{epoch+1}/{total_epochs}], \
            Loss: {loss_value.item():.4f}', end="")

    if time is None:
        print("")
    else:
        print(f", Elapsed Time: {time:.2f} sec")
