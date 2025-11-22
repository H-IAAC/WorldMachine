import numpy as np
import torch
from tensordict import TensorDict
from torch.nn import Module, ModuleDict
from torch.optim import Optimizer

from world_machine.layers import PointwiseFeedforward
from world_machine.world_machine import WorldMachine

from .train_stage import TrainStage


class ShortTimeRecaller(TrainStage):
    def __init__(self, channel_sizes: dict[str, int], criterions: dict[str, Module], n_past: int = 0, n_future: int = 0,
                 stride_past: int = 1, stride_future: int = 1):
        super().__init__(-2)

        self._channels = channel_sizes
        self._criterions = criterions

        self._projectors: ModuleDict = ModuleDict()

        self._n_past = n_past
        self._n_future = n_future

        self._stride_past = stride_past
        self._stride_future = stride_future

        self._original_param_group = []

    def pre_train(self,
                  model: WorldMachine,
                  criterions: dict[str, dict[str, Module]],
                  train_criterions: dict[str, dict[str, float]],
                  device: torch.device,
                  optimizer: Optimizer) -> None:

        self._original_param_group = list(optimizer.param_groups)

        state_size = model._state_size

        # weights = np.exp(-np.linspace(0, self._n_past-1, self._n_past))
        weights_past = np.linspace(self._n_past, 1, self._n_past)
        weights_future = np.linspace(self._n_future, 1, self._n_future)

        total = weights_past.sum()+weights_future.sum()

        weights_past /= total
        weights_future /= total

        for channel in self._channels:
            channel_size = self._channels[channel]

            self._projectors[channel] = torch.nn.Linear(
                channel_size, channel_size).to(device)
            self._projectors[channel].eval()

            for name, n, weights in zip(["past", "future"], [self._n_past, self._n_future], [weights_past, weights_future]):
                for i in range(n):
                    channel_name = f"{name}{i}_{channel}"

                    decoder = PointwiseFeedforward(state_size,
                                                   state_size*2,
                                                   output_dim=channel_size).to(device)

                    model._sensory_decoders[channel_name] = decoder
                    optimizer.add_param_group({"params": decoder.parameters()})

                    criterions[channel_name] = {
                        "loss": self._criterions[channel]}

                    train_criterions[channel_name] = {"loss": weights[i]}

    def pre_segment(self, itens, losses, batch_size, seq_len, epoch_index, device, state_size, mode, model: WorldMachine):

        item = itens[0]

        if "target_masks" not in item:
            item["target_masks"] = TensorDict(
                batch_size=item["targets"].batch_size)

        with torch.no_grad():
            for channel in self._channels:
                data: torch.Tensor = item["targets"][channel]

                if channel in item["target_masks"]:
                    mask = item["target_masks"][channel]
                else:
                    mask = torch.ones([batch_size, seq_len],
                                      dtype=bool, device=device)

                for i in range(self._n_future):
                    future_channel_name = f"future{i}_{channel}"
                    # i=0 is itself, i=1 is the same as normal train
                    future_index = (i*self._stride_future)+2

                    future_data = torch.roll(data, -future_index, 1)
                    future_mask = torch.roll(mask, -future_index, 1)

                    future_data: torch.Tensor = self._projectors[channel](
                        future_data).detach()
                    future_mask[:, future_mask.shape[0]-i:] = False

                    item["targets"][future_channel_name] = future_data

                    item["target_masks"][future_channel_name] = future_mask

                for i in range(self._n_past):
                    past_channel_name = f"past{i}_{channel}"
                    past_index = -(i*self._stride_past)-1

                    past_data = torch.roll(data, -past_index, 1)
                    past_mask = torch.roll(mask, -past_index, 1)

                    past_data: torch.Tensor = self._projectors[channel](
                        past_data).detach()
                    past_mask[:, :i+1] = False

                    item["targets"][past_channel_name] = past_data
                    item["target_masks"][past_channel_name] = past_mask

    def post_train(self, model: WorldMachine,
                   criterions: dict[str, dict[str, Module]],
                   train_criterions: dict[str, dict[str, float]],
                   optimizer: Optimizer) -> None:
        for channel in self._channels:
            for name, n in zip(["past", "future"], [self._n_past, self._n_future]):
                for i in range(n):
                    channel_name = f"{name}{i}_{channel}"

                    del model._sensory_decoders[channel_name]

                    del criterions[channel_name]
                    del train_criterions[channel_name]

        optimizer.state.clear()
        optimizer.param_groups = self._original_param_group
