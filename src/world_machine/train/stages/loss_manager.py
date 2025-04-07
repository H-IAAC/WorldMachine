import torch
from tensordict import TensorDict
from torch.nn import Module
from torch.optim import Optimizer

from world_machine import WorldMachine
from world_machine.data import WorldMachineDataset
from world_machine.train import DatasetPassMode

from .train_stage import TrainStage


class LossManager(TrainStage):
    def __init__(self):
        super().__init__(2)
        self.n: int

    def pre_batch(self, model: WorldMachine, mode: DatasetPassMode,
                  criterions: dict[str, dict[str, Module]], optimizer: Optimizer,
                  device: torch.device, losses: dict) -> None:
        total_loss: dict[str, dict[str, torch.Tensor] | torch.Tensor] = {}
        for dimension in criterions:
            total_loss[dimension] = {}
            for criterion_name in criterions[dimension]:
                total_loss[dimension][criterion_name] = torch.tensor(
                    0, dtype=torch.float32, device=device)

        total_loss["optimizer_loss"] = torch.tensor(
            0, dtype=torch.float32, device=device)

        losses.clear()
        losses["epoch"] = total_loss

        self.n = 0

    def post_batch(self, model: WorldMachine, losses: dict) -> None:
        total_loss = losses["epoch"]

        result = {}
        for dimension in total_loss:
            if dimension == "optimizer_loss":
                result[dimension] = total_loss[dimension]
            else:
                for criterion_name in total_loss[dimension]:
                    result[f"{dimension}_{criterion_name}"] = total_loss[dimension][criterion_name]

        losses.clear()
        losses.update(result)

    def post_segment(self, itens: list[TensorDict], losses: dict, dataset: WorldMachineDataset, epoch_index: int,
                     criterions: dict[str, dict[str, Module]], mode: DatasetPassMode, device: torch.device, train_criterions: dict[str, dict[str, float]]) -> None:

        item = itens[0]
        targets = item["targets"]
        logits = item["logits"]

        targets_masks = None
        if "target_masks" in item:
            targets_masks = item["target_masks"]

        total_loss = losses["epoch"]

        item_losses: dict[str, dict[str, torch.Tensor] | torch.Tensor] = {}
        for dimension in criterions:
            if len(criterions[dimension]) == 0:
                continue

            logits_dim = logits[dimension]
            targets_dim = targets[dimension]

            if (targets_masks is not None and
                    dimension in targets_masks):

                logits_dim = logits_dim[targets_masks[dimension]]
                targets_dim = targets_dim[targets_masks[dimension]]

            item_losses[dimension] = {}
            for criterion_name in criterions[dimension]:
                if criterion_name not in train_criterions[dimension]:
                    torch.set_grad_enabled(False)

                item_losses[dimension][criterion_name] = criterions[dimension][criterion_name](
                    logits_dim, targets_dim)

                total_loss[dimension][criterion_name] += item_losses[dimension][criterion_name] * \
                    targets.size(0)

                torch.set_grad_enabled(
                    mode == DatasetPassMode.MODE_TRAIN)

        optimizer_loss = torch.tensor(
            0, dtype=torch.float32, device=device)
        for dimension in train_criterions:
            for criterion_name in train_criterions[dimension]:
                optimizer_loss += item_losses[dimension][criterion_name] * \
                    train_criterions[dimension][criterion_name]

        item_losses["optimizer_loss"] = optimizer_loss

        total_loss["optimizer_loss"] += item_losses["optimizer_loss"] * \
            targets.size(0)
        self.n += targets.size(0)

        losses["optimizer_loss"] = optimizer_loss
