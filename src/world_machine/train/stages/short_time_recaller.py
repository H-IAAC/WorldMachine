import numpy as np
import torch
from tensordict import TensorDict
from torch.nn import Module, ModuleDict
from torch.optim import Optimizer

from world_machine import WorldMachine
from world_machine.layers import PointwiseFeedforward
from world_machine.train import DatasetPassMode

from .train_stage import TrainStage


class ShortTimeRecaller(TrainStage):
    def __init__(self, dimension_sizes: dict[str, int], criterions: dict[str, Module], n_past: int = 0, n_future: int = 0,
                 stride_past: int = 1, stride_future: int = 1):
        super().__init__(-1)

        self._dimensions = dimension_sizes
        self._criterions = criterions

        self._projectors: ModuleDict = ModuleDict()

        self._n_past = n_past
        self._n_future = n_future

        self._stride_past = stride_past
        self._stride_future = stride_future

    def pre_train(self, model: WorldMachine, criterions: dict[str, dict[str, Module]],  train_criterions: dict[str, dict[str, float]], device: torch.device) -> None:

        state_size = model._state_size

        # weights = np.exp(-np.linspace(0, self._n_past-1, self._n_past))
        weights_past = np.linspace(self._n_past, 1, self._n_past)
        weights_future = np.linspace(self._n_future, 1, self._n_future)

        total = weights_past.sum()+weights_future.sum()

        weights_past /= total
        weights_future /= total

        for dimension in self._dimensions:
            dimension_size = self._dimensions[dimension]

            self._projectors[dimension] = torch.nn.Linear(
                dimension_size, dimension_size).to(device)
            self._projectors[dimension].eval()

            for name, n, weights in zip(["past", "future"], [self._n_past, self._n_future], [weights_past, weights_future]):
                for i in range(n):
                    dim_name = f"{name}{i}_{dimension}"

                    model._sensorial_decoders[dim_name] = PointwiseFeedforward(
                        state_size, state_size*2, output_dim=dimension_size).to(device)

                    criterions[dim_name] = {
                        "loss": self._criterions[dimension]}

                    train_criterions[dim_name] = {"loss": weights[i]}

    def pre_segment(self, itens, losses, batch_size, seq_len, epoch_index, device, state_size, mode):

        item = itens[0]

        if "target_masks" not in item:
            item["target_masks"] = TensorDict(
                batch_size=item["targets"].batch_size)

        with torch.no_grad():
            for dimension in self._dimensions:
                data: torch.Tensor = item["targets"][dimension]

                if dimension in item["target_masks"]:
                    mask = item["target_masks"][dimension]
                else:
                    mask = torch.ones([batch_size, seq_len],
                                      dtype=bool, device=device)

                for i in range(self._n_future):
                    future_dim_name = f"future{i}_{dimension}"
                    # i=0 is itself, i=1 is the same as normal train
                    future_index = (i*self._stride_future)+2

                    future_data = torch.roll(data, -future_index, 1)
                    future_mask = torch.roll(mask, -future_index, 1)

                    future_data: torch.Tensor = self._projectors[dimension](
                        future_data).detach()
                    future_mask[:, future_mask.shape[0]-i:] = False

                    item["targets"][future_dim_name] = future_data

                    item["target_masks"][future_dim_name] = future_mask

                for i in range(self._n_past):
                    past_dim_name = f"past{i}_{dimension}"
                    past_index = -(i*self._stride_past)-1

                    past_data = torch.roll(data, -past_index, 1)
                    past_mask = torch.roll(mask, -past_index, 1)

                    past_data: torch.Tensor = self._projectors[dimension](
                        past_data).detach()
                    past_mask[:, :i+1] = False

                    item["targets"][past_dim_name] = past_data
                    item["target_masks"][past_dim_name] = past_mask

    def post_train(self, model: WorldMachine, criterions: dict[str, dict[str, Module]], train_criterions: dict[str, dict[str, float]]) -> None:
        for dimension in self._dimensions:
            for name, n in zip(["past", "future"], [self._n_past, self._n_future]):
                for i in range(n):
                    dim_name = f"{name}{i}_{dimension}"

                    del model._sensorial_decoders[dim_name]

                    del criterions[dim_name]
                    del train_criterions[dim_name]
