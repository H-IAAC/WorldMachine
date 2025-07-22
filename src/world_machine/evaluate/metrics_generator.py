import tensordict
import torch
from tensordict import TensorDict

from world_machine.data import WorldMachineDataLoader, WorldMachineDataset
from world_machine.train import CriterionSet, DatasetPassMode
from world_machine.train.stages import LossManager
from world_machine.world_machine import WorldMachine


class MetricsGenerator:

    def __init__(self, criterion_set: CriterionSet):
        self._criterion_set = criterion_set

    def _inference(self,
                   model: WorldMachine,
                   data_loader: WorldMachineDataLoader,
                   batch_size: int,
                   seq_len: int) -> tuple[TensorDict, torch.Tensor]:

        state_size = model._state_size

        device = next(iter(model.parameters())).device

        logits: list[TensorDict] = []
        states: list[torch.Tensor] = []
        for item in data_loader:
            item: TensorDict
            inputs: torch.Tensor = item["inputs"].to(device)

            states_batch = torch.empty(
                [batch_size, seq_len, state_size], device=device)
            states_batch[:, 0, :] = 0

            logits_batch = model.inference(
                states_batch, inputs, total_size=inputs.shape[1])
            states_batch[:, 1:] = logits_batch["state"][:, :-1]

            logits.append(logits_batch)
            states.append(states_batch)

        logits = tensordict.lazy_stack(logits)
        states = torch.stack(states)

        return logits, states

    def _inference_previous_coded(self,
                                  model: WorldMachine,
                                  data_loader: WorldMachineDataLoader,
                                  states: torch.Tensor,
                                  sensorial_masks: TensorDict | None = None,
                                  inference_start: int = 0,
                                  data_start: int = 0,
                                  replace_sensorial_data: bool = True) -> TensorDict:

        device = next(iter(model.parameters())).device

        logits: list[TensorDict] = []
        index = 0
        for item in data_loader:
            item: TensorDict
            inputs: torch.Tensor = item["inputs"].to(device)
            states_batch = states[index].to(device)

            logits_batch = model.inference(states_batch[:, data_start:],
                                           inputs[:, data_start:],
                                           sensorial_masks[:, data_start:],
                                           start=inference_start,
                                           replace_sensorial_data=replace_sensorial_data)

            logits.append(logits_batch)

        logits = tensordict.lazy_stack(logits)

        return logits

    def _generate_masked_masks(self, inputs: TensorDict) -> TensorDict:
        batch_size = inputs.batch_size[0]
        seq_len = inputs[next(
            iter(inputs.keys()))].shape[1]
        device = inputs.device

        sensorial_masks_masked = TensorDict(
            device=device, batch_size=[batch_size, seq_len])

        sensorial_data: TensorDict = inputs
        for name in sensorial_data.keys():
            sensorial_masks_masked[name] = torch.zeros(
                (batch_size, seq_len), dtype=bool, device=device)

        return sensorial_masks_masked

    def __call__(self, model: WorldMachine, dataset: WorldMachineDataset, batch_size: int):
        original_grad_state = torch.is_grad_enabled()
        original_model_state = model.training
        torch.set_grad_enabled(False)

        # Must not shuffle (sync logits and targets)
        data_loader = WorldMachineDataLoader(
            dataset, batch_size, shuffle=False)

        # Prepare Data
        item = next(iter(data_loader))
        batch_size = item["inputs"].batch_size[0]
        seq_len = item["inputs"][next(
            iter(item["inputs"].keys()))].shape[1]
        sensorial_masks_masked = self._generate_masked_masks(item["inputs"])
        del item

        device = next(iter(model.parameters())).device

        half_seq_len = seq_len//2

        # Compute logits
        logits, states = self._inference(
            model, data_loader, batch_size, seq_len)
        logits_use_pred = self._inference_previous_coded(
            model, data_loader, states, sensorial_masks_masked, inference_start=half_seq_len)
        logits_pred_shallow = self._inference_previous_coded(
            model, data_loader, states, sensorial_masks_masked, data_start=half_seq_len)

        logits_use_state = logits_use_pred[:, :half_seq_len]
        logits_prediction = logits[:, half_seq_len:]

        loss_manager = LossManager()

        all_logits = {"normal": logits,
                      "use_state": logits_use_state,
                      "prediction": logits_prediction,
                      "prediction_shallow": logits_pred_shallow}

        all_losses = {}

        for name in all_logits:
            losses = {}
            loss_manager.pre_batch(model,
                                   DatasetPassMode.MODE_EVALUATE,
                                   self._criterion_set.criterions,
                                   None,
                                   device,
                                   losses,
                                   self._criterion_set.train_criterions)

            for batch_index, item in enumerate(data_loader):
                item["logits"] = all_logits[name][batch_index]
                itens = [item]

                loss_manager.post_segment(itens,
                                          losses,
                                          dataset,
                                          0,
                                          self._criterion_set.criterions,
                                          DatasetPassMode.MODE_EVALUATE,
                                          device,
                                          self._criterion_set.train_criterions)

            loss_manager.post_batch(
                model, losses, self._criterion_set.criterions, self._criterion_set.train_criterions)

            all_losses[name] = losses

        torch.set_grad_enabled(original_grad_state)
        if original_model_state:
            model.train()
        else:
            model.eval()

        return all_losses
