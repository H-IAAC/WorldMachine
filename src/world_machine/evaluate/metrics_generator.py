import torch
import tqdm
from tensordict import TensorDict

from world_machine.data import WorldMachineDataLoader
from world_machine.profile import profile_range
from world_machine.train import CriterionSet, DatasetPassMode
from world_machine.train.stages import LossManager, PrepareModel, TrainStage
from world_machine.world_machine import WorldMachine


class MetricsGenerator:

    def __init__(self, criterion_set: CriterionSet):
        self._criterion_set = criterion_set

        self.stages: list[TrainStage] = []

    @profile_range("inference", category="metrics", domain="world_machine")
    def _inference(self,
                   model: WorldMachine,
                   item: TensorDict,
                   batch_size: int,
                   seq_len: int) -> tuple[TensorDict, torch.Tensor]:

        state_size = model._state_size

        device = next(iter(model.parameters())).device

        inputs: torch.Tensor = item["inputs"].to(device)

        masks = None
        if "input_masks" in item:
            masks = item["input_masks"]

        state = torch.empty(
            [batch_size, seq_len, state_size], device=device)
        state[:, 0, :] = 0

        logits = model.inference(
            state, inputs, masks, total_size=inputs.shape[1])
        state[:, 1:] = logits["state"][:, :-1]

        return logits, state

    @profile_range("inference_previous_coded", category="metrics", domain="world_machine")
    def _inference_previous_coded(self,
                                  model: WorldMachine,
                                  item: TensorDict,
                                  state: torch.Tensor,
                                  sensorial_masks: TensorDict | None = None,
                                  inference_start: int = 0,
                                  data_start: int = 0,
                                  replace_sensorial_data: bool = True) -> TensorDict:

        device = next(iter(model.parameters())).device

        inputs: torch.Tensor = item["inputs"].to(device)

        logits = model.inference(state[:, data_start:],
                                 inputs[:, data_start:],
                                 sensorial_masks[:, data_start:],
                                 start=inference_start,
                                 replace_sensorial_data=replace_sensorial_data)

        return logits

    @profile_range("generate_masked_masks", category="metrics", domain="world_machine")
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

    @profile_range("metrics_generator_call", category="metrics", domain="world_machine")
    def __call__(self, model: WorldMachine, dataloader: WorldMachineDataLoader,
                 return_logits: bool = False,
                 compute_use_state: bool = True,
                 compute_prediction: bool = True,
                 compute_prediction_shallow: bool = True,
                 ):
        dataset = dataloader.dataset

        # Prepare Data
        item = next(iter(dataloader))
        batch_size = item["inputs"].batch_size[0]
        seq_len = item["inputs"][next(
            iter(item["inputs"].keys()))].shape[1]
        sensorial_masks_masked = self._generate_masked_masks(item["inputs"])
        del item

        device = next(iter(model.parameters())).device

        half_seq_len = seq_len//2

        loss_manager = LossManager()
        prepare_model = PrepareModel()

        names = ["normal"]
        if compute_use_state:
            names.append("use_state")
        if compute_prediction:
            names.append("prediction")
        if compute_prediction_shallow:
            names.append("prediction_shallow")

        all_losses: dict[str, TensorDict] = {}
        all_logits: dict[str, list[TensorDict]] = {}
        for name in names:
            all_losses[name] = {}
            loss_manager.pre_batch(model,
                                   DatasetPassMode.MODE_EVALUATE,
                                   self._criterion_set.criterions,
                                   None,
                                   device,
                                   all_losses[name],
                                   self._criterion_set.train_criterions)

            all_logits[name] = []

        prepare_model.pre_batch(model,
                                DatasetPassMode.MODE_EVALUATE,
                                self._criterion_set.criterions,
                                None,
                                device,
                                None,
                                self._criterion_set.train_criterions)

        for item in tqdm.tqdm(dataloader, desc="Metrics Generation"):
            item = item.to(device)

            del item["index"]
            item.batch_size = [batch_size, seq_len]

            for stage in self.stages:
                stage.pre_segment([item], None, batch_size, seq_len, 0,
                                  device, model._state_size, DatasetPassMode.MODE_EVALUATE, model)

            logits, state = self._inference(model,
                                            item,
                                            batch_size,
                                            seq_len)

            item["logits"] = logits

            if return_logits:
                all_logits["normal"].append(item["logits"].cpu())

            loss_manager.post_segment([item],
                                      all_losses["normal"],
                                      dataset,
                                      0,
                                      self._criterion_set.criterions,
                                      DatasetPassMode.MODE_EVALUATE,
                                      device,
                                      self._criterion_set.train_criterions)

            if compute_use_state or compute_prediction or compute_prediction_shallow:
                item["logits"] = self._inference_previous_coded(model,
                                                                item,
                                                                state,
                                                                sensorial_masks_masked,
                                                                inference_start=half_seq_len)

                itens: dict[str, list[TensorDict]] = {}
                if compute_use_state:
                    itens["use_state"] = [item[:, :half_seq_len]]
                if compute_prediction:
                    itens["prediction"] = [item[:, half_seq_len:]]

                for name in itens:
                    loss_manager.post_segment(itens[name],
                                              all_losses[name],
                                              dataset,
                                              0,
                                              self._criterion_set.criterions,
                                              DatasetPassMode.MODE_EVALUATE,
                                              device,
                                              self._criterion_set.train_criterions)

                del item["logits"]

                if compute_prediction_shallow:

                    logits_pred_shallow = self._inference_previous_coded(model,
                                                                         item,
                                                                         state,
                                                                         sensorial_masks_masked,
                                                                         data_start=half_seq_len)

                    item = item[:, half_seq_len:]
                    item["logits"] = logits_pred_shallow

                    loss_manager.post_segment([item],
                                              all_losses["prediction_shallow"],
                                              dataset,
                                              0,
                                              self._criterion_set.criterions,
                                              DatasetPassMode.MODE_EVALUATE,
                                              device,
                                              self._criterion_set.train_criterions)

                if return_logits:
                    all_logits["use_state"].append(
                        itens["use_state"][0]["logits"].cpu())
                    all_logits["prediction"].append(
                        itens["prediction"][0]["logits"].cpu())

                    if compute_prediction_shallow:
                        all_logits["prediction_shallow"].append(
                            logits_pred_shallow.cpu())

        for name in all_losses:
            loss_manager.post_batch(
                model, all_losses[name], self._criterion_set.criterions, self._criterion_set.train_criterions)

        prepare_model.post_batch(
            model, all_losses[name], self._criterion_set.criterions, self._criterion_set.train_criterions)

        for name in all_losses:
            for loss_name in all_losses[name]:
                all_losses[name][loss_name] = all_losses[name][loss_name].cpu().item()

        if return_logits:
            for name in all_logits:
                all_logits[name] = torch.cat(all_logits[name], 0)

            return all_losses, all_logits
        return all_losses
