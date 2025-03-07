import numpy as np
import torch
from tensordict import TensorDict

from world_machine.data import WorldMachineDataset


class Toy1dDataset(WorldMachineDataset):
    def __init__(self, data: dict[str, np.ndarray], context_size: int,
                 return_state_dimensions: list[int] | None = None):

        self._data = data
        self._context_size = context_size
        self._return_dimensions = return_state_dimensions

        self._n_sequence = self._data["state_decoded"].shape[0]
        self._sequence_size = self._data["state_decoded"].shape[1]
        self._items_in_sequence = (self._sequence_size-1)//self._context_size

        size = self._n_sequence*int((self._sequence_size-1)/context_size)

        super().__init__(
            ["state_control", "next_measurement"],
            size=size,
            state_size=6,
            has_state_decoded=True,
            has_masks=False)

    def get_dimension_item(self, dimension: str, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item_index = index // self._items_in_sequence
        item_seq_index = index % self._items_in_sequence

        start = item_seq_index*self._context_size
        end = start+self._context_size

        item = []

        for i in range(2):
            item.append(torch.Tensor(
                self._data[dimension][item_index, start+i:end+i]))

            if self._return_dimensions is not None and dimension == "state_decoded":
                item[i] = item[i][:, self._return_dimensions]

        return item


def toy1d_datasets(toy1d_data_splitted: dict[str, dict[str, np.ndarray]], context_size: int,
                   state_dimensions: list[int] | None = None) -> dict[str, Toy1dDataset]:

    datasets = {}
    for name in ["train", "val", "test"]:
        datasets[name] = Toy1dDataset(toy1d_data_splitted[name],
                                      context_size=context_size,
                                      return_state_dimensions=state_dimensions)

    return datasets
