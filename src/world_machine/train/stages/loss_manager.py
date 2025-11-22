import torch
from tensordict import TensorDict
from torch.nn import Module
from torch.optim import Optimizer

from world_machine.data import WorldMachineDataset
from world_machine.train.mode import DatasetPassMode
from world_machine.world_machine import WorldMachine

from .train_stage import TrainStage


class LossManager(TrainStage):
    def __init__(self,
                 state_regularizer: str | None = None,
                 state_cov_regularizer: float | None = None,
                 multiply_target_masks: bool = True):
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
        self._multiply_target_masks = multiply_target_masks

    def pre_batch(self, model: WorldMachine, mode: DatasetPassMode,
                  criterions: dict[str, dict[str, Module]], optimizer: Optimizer,
                  device: torch.device, losses: dict, train_criterions: dict[str, dict[str, float]]) -> None:
        total_loss: dict[str, dict[str, torch.Tensor] | torch.Tensor] = {}
        for channel in criterions:
            total_loss[channel] = {}
            for criterion_name in criterions[channel]:
                total_loss[channel][criterion_name] = torch.tensor(
                    0, dtype=torch.float32, device=device)

        total_loss["optimizer_loss"] = torch.tensor(
            0, dtype=torch.float32, device=device)

        losses.clear()
        losses["epoch"] = total_loss

        self.n = 0

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
        for channel in criterions:
            if len(criterions[channel]) == 0:
                continue

            logits_channel = logits[channel]
            targets_channel = targets[channel]

            mask_factor = 1.0

            if (targets_masks is not None and
                    channel in targets_masks):

                if self._multiply_target_masks:
                    logits_channel *= targets_masks[channel].unsqueeze(2)
                    targets_channel *= targets_masks[channel].unsqueeze(2)

                    mask_factor = (
                        targets_masks[channel].numel() / targets_masks[channel].sum())
                else:
                    logits_channel = logits_channel[:,
                                                    targets_masks[channel][0]]
                    targets_channel = targets_channel[:,
                                                      targets_masks[channel][0]]

            item_losses[channel] = {}
            for criterion_name in criterions[channel]:
                if criterion_name not in train_criterions[channel]:
                    torch.set_grad_enabled(False)

                item_losses[channel][criterion_name] = criterions[channel][criterion_name](
                    logits_channel, targets_channel) * mask_factor

                total_loss[channel][criterion_name] += item_losses[channel][criterion_name] * \
                    targets.size(0)

                torch.set_grad_enabled(
                    mode == DatasetPassMode.MODE_TRAIN)

        optimizer_loss = torch.tensor(
            0, dtype=torch.float32, device=device)
        total_weight = 0

        for channel in train_criterions:
            for criterion_name in train_criterions[channel]:
                optimizer_loss += item_losses[channel][criterion_name] * \
                    train_criterions[channel][criterion_name]

                total_weight += train_criterions[channel][criterion_name]

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

    def post_batch(self,
                   model: WorldMachine,
                   losses: dict,
                   criterions: dict[str, dict[str, Module]],
                   train_criterions: dict[str, dict[str, float]],
                   mode: DatasetPassMode) -> None:
        total_loss = losses["epoch"]

        for channel in total_loss:
            if channel == "optimizer_loss":
                total_loss[channel] /= self.n
                total_loss[channel] = total_loss[channel].detach()
            else:
                for criterion_name in total_loss[channel]:
                    total_loss[channel][criterion_name] /= self.n
                    total_loss[channel][criterion_name] = total_loss[channel][criterion_name].detach(
                    )

        result = {}
        for channel in total_loss:
            if channel == "optimizer_loss":
                result[channel] = total_loss[channel]
            else:
                for criterion_name in total_loss[channel]:
                    result[f"{channel}_{criterion_name}"] = total_loss[channel][criterion_name]

        losses.clear()
        losses.update(result)
