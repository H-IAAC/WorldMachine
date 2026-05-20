import numpy as np
import tensordict
import torch
from tensordict import TensorDict
from torch.nn import Module
from torch.optim import Optimizer

from world_machine.data import (
    DatasetItem, IndexedDatasetItem, WorldMachineDataset)
from world_machine.train.mode import DatasetPassMode
from world_machine.world_machine import WorldMachine

from .train_stage import TrainStage


class SequenceBreaker(TrainStage):
    def __init__(self, n_segment: int = 1, fast_forward: bool = False):
        super().__init__(4)

        self._n_point = n_segment-1
        self._fast_forward = fast_forward

    def pre_segment(self, itens: list[IndexedDatasetItem], losses: dict, batch_size: int,
                    seq_len: int, epoch_index: int, device: torch.device,
                    state_size: int, mode: DatasetPassMode, model: WorldMachine) -> None:

        if mode == DatasetPassMode.MODE_TRAIN:
            indexes = self.np_generator.choice(np.linspace(
                0, seq_len-1, seq_len, dtype=int), size=self._n_point, replace=False)

            indexes = np.append(indexes, [seq_len])
            indexes = np.delete(indexes, np.argwhere(indexes == 0))
            indexes.sort()

            sizes = np.append(indexes[0], indexes[1:] - indexes[:-1])

            item = itens[0]

            index = item.index
            item = item.deindex()
            item.batch_size = [batch_size, seq_len]

            segments: list[DatasetItem] = item.split(
                sizes.tolist(), dim=1)

            indexed_segments: list[IndexedDatasetItem] = []
            for segment in segments:
                segment.batch_size = [batch_size]
                segment = segment.index_item(index)

                indexed_segments.append(segment)

            item.batch_size = [batch_size]
            indexed_item = item.index_item(index)

            itens.clear()
            itens.extend(indexed_segments)

    def post_segment(self, itens: list[IndexedDatasetItem], losses: dict, dataset: WorldMachineDataset, epoch_index: int,
                     criterions: dict[str, dict[str, Module]], mode: DatasetPassMode,
                     device: torch.device, train_criterions: dict[str, dict[str, float]]) -> None:
        if mode == DatasetPassMode.MODE_TRAIN:
            index = itens[0]["index"]
            batch_size = itens[0].batch_size[0]

            deindexed_itens: list[DatasetItem] = []
            for item in itens:
                item = item.deindex()
                seq_len = item.inputs[next(
                    iter(item.inputs.keys()))].shape[1]
                item.batch_size = [batch_size, seq_len]

                deindexed_itens.append(item)

            reconstructed_item: DatasetItem = tensordict.cat(
                deindexed_itens, dim=1)
            reconstructed_item.batch_size = [batch_size]

            indexed_reconstructed_item = reconstructed_item.index_item(index)

            itens.clear()
            itens.append(indexed_reconstructed_item)

    def post_forward(self, item_index: int,  itens: list[IndexedDatasetItem], dataset: WorldMachineDataset, losses: dict, mode: DatasetPassMode) -> None:
        if mode == DatasetPassMode.MODE_TRAIN and self._fast_forward and item_index+1 < len(itens) and "state" in itens[0].inputs:
            current_item = itens[item_index]
            next_item = itens[item_index+1]

            assert (current_item.logits is not None)
            statenext_current = current_item.logits["state"]

            state = next_item.inputs["state"]
            state = torch.cat(
                (statenext_current[:, -1].unsqueeze(1), state[:, 1:]),
                dim=1
            )

            next_item.inputs["state"] = state
