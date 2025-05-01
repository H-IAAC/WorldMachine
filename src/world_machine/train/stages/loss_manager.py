import torch
from tensordict import TensorDict
from torch.nn import Module
from torch.optim import Optimizer

from world_machine import WorldMachine
from world_machine.data import WorldMachineDataset
from world_machine.train import DatasetPassMode

from .train_stage import TrainStage


class LossManager(TrainStage):
    def __init__(self, state_regularizer: str | None = None, state_cov_regularizer: float | None = None):
        super().__init__(2)
        self.n: int

        if state_regularizer is None:
            self._state_regularizer = state_regularizer
        elif state_regularizer == "mse":
            self._state_regularizer = torch.nn.MSELoss()
        else:
            raise ValueError(
                f"state_regularizer mode {state_regularizer} not valid.")

        self._state_cov_regularizer = state_cov_regularizer

    def pre_batch(self, model: WorldMachine, mode: DatasetPassMode,
                  criterions: dict[str, dict[str, Module]], optimizer: Optimizer,
                  device: torch.device, losses: dict, train_criterions: dict[str, dict[str, float]]) -> None:
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

    def post_batch(self, model: WorldMachine, losses: dict, criterions: dict[str, dict[str, Module]], train_criterions: dict[str, dict[str, float]]) -> None:
        total_loss = losses["epoch"]

        for dimension in total_loss:
            if dimension == "optimizer_loss":
                total_loss[dimension] /= self.n
                total_loss[dimension] = total_loss[dimension].detach()
            else:
                for criterion_name in total_loss[dimension]:
                    total_loss[dimension][criterion_name] /= self.n
                    total_loss[dimension][criterion_name] = total_loss[dimension][criterion_name].detach(
                    )

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

                logits_dim = logits_dim[:, targets_masks[dimension][0]]
                targets_dim = targets_dim[:, targets_masks[dimension][0]]

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
        total_weight = 0

        for dimension in train_criterions:
            for criterion_name in train_criterions[dimension]:
                optimizer_loss += item_losses[dimension][criterion_name] * \
                    train_criterions[dimension][criterion_name]

                total_weight += train_criterions[dimension][criterion_name]

        optimizer_loss /= total_weight

        if self._state_regularizer is not None:
            optimizer_loss += 0.5*self._state_regularizer(
                logits["state"], torch.zeros_like(logits["state"]))

        if self._state_cov_regularizer is not None:
            batch_size = itens[0].batch_size[0]
            cov_sum = torch.empty(batch_size)

            for i in range(batch_size):
                cov_sum[i] = torch.pow(torch.tril(
                    torch.cov(logits["state"][i].T), diagonal=0), 2).mean()

            mean_state_cov = cov_sum.mean()

            optimizer_loss += self._state_cov_regularizer*(-mean_state_cov)

        item_losses["optimizer_loss"] = optimizer_loss

        total_loss["optimizer_loss"] += item_losses["optimizer_loss"] * \
            targets.size(0)
        self.n += targets.size(0)

        losses["optimizer_loss"] = optimizer_loss
