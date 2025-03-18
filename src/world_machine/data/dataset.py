import abc
import atexit
import os
from collections import deque

import torch
from tensordict import MemoryMappedTensor, TensorDict
from torch.utils.data import DataLoader, Dataset


class WorldMachineDataset(Dataset, abc.ABC):

    _states_filenames = deque()

    def __init__(self, sensorial_dimensions: list[str], size: int,
                 state_size: int,
                 has_state_decoded: bool = False,
                 has_masks: bool = False):
        super().__init__()

        self._sensorial_dimensions = sensorial_dimensions
        self._size = size
        self._state_size = state_size
        self._has_state_decoded = has_state_decoded
        self._has_masks = has_masks

        self._states = None
        self._states_filename = None

    def __len__(self) -> int:
        return self._size

    def load_data(self, index: int) -> None:
        ...

    @abc.abstractmethod
    def get_dimension_item(self, dimension: str, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def get_dimension_mask(self, dimension: str, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def __getitem__(self, index):
        item = TensorDict(
            {"inputs": TensorDict(), "targets": TensorDict(), "index": index}, batch_size=[])

        self.load_data(index)
        for dimension in self._sensorial_dimensions:
            item["inputs"][dimension], item["targets"][dimension] = self.get_dimension_item(
                dimension, index)

        if self._has_state_decoded:
            item["inputs"]["state_decoded"], item["targets"]["state_decoded"] = self.get_dimension_item(
                "state_decoded", index)

        if self._has_masks:
            for dimension in self._sensorial_dimensions:
                (item["inputs"]["sensorial_masks"][dimension],
                 item["targets"]["sensorial_masks"][dimension]) = self.get_dimension_mask(dimension, index)

        if self._states != None:
            item["inputs"]["state"] = self._states[index]

        self.dispose_data(index)
        return item

    def dispose_data(self, index: int) -> None:
        ...

    def set_state(self, indexes: int, states: torch.Tensor) -> None:
        if self._states is None:
            dtype = states.dtype
            states_shape = list(states.shape)
            states_shape[0] = self._size

            i = 0
            while True:
                filename = f"TempStates_{self.__class__.__name__}_{i}.bin"

                if not os.path.exists(filename):
                    break

                i += 1

            self._states = MemoryMappedTensor.empty(
                states_shape, dtype=dtype, filename=filename)

            self._states_filename = filename
            WorldMachineDataset._states_filenames.append(filename)

        self._states[indexes.cpu()] = states.detach().cpu()

    @classmethod
    def _delete_files(cls):
        for filename in cls._states_filenames:
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass


atexit.register(WorldMachineDataset._delete_files)
