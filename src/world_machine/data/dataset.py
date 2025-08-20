import abc
import atexit
import os
from collections import deque

import torch
from tensordict import MemoryMappedTensor, TensorDict
from torch.utils.data import DataLoader, Dataset

from world_machine.profile import profile_range


class WorldMachineDataset(Dataset, abc.ABC):

    _states_filenames = deque()

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)

        name = cls.__name__

        methods = {"load_data": instance.load_data,
                   "get_dimension_item": instance.get_dimension_item,
                   "get_dimension_mask": instance.get_dimension_mask,
                   "dispose_data": instance.dispose_data}

        for method_name in methods:
            if method_name in cls.__dict__:
                method = methods[method_name]
                method = profile_range(
                    f"{name}_{method_name}", category="wm_dataset", domain="world_machine")(method)

                instance.__setattr__(method_name, method)

        return instance

    def __init__(self, sensorial_dimensions: list[str], size: int,
                 has_state_decoded: bool = False,
                 has_masks: bool = False,
                 map_state_to_disk: bool = True):
        super().__init__()

        self._sensorial_dimensions = sensorial_dimensions
        self._size = size
        self._has_state_decoded = has_state_decoded
        self._has_masks = has_masks

        self._map_state_to_disk = map_state_to_disk
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

    @profile_range("__getitem__", category="wm_dataset", domain="world_machine")
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

        if self._states != None:
            item["inputs"]["state"] = self._states[index]

        seq_len = item["inputs"][next(iter(item["inputs"].keys()))].shape[0]

        item["inputs"].batch_size = [seq_len]
        item["targets"].batch_size = [seq_len]

        if self._has_masks:
            item["input_masks"] = TensorDict()
            item["target_masks"] = TensorDict()

            for dimension in self._sensorial_dimensions:
                (item["input_masks"][dimension],
                 item["target_masks"][dimension]) = self.get_dimension_mask(dimension, index)

            if self._has_state_decoded:
                (item["input_masks"]["state_decoded"],
                 item["target_masks"]["state_decoded"]) = self.get_dimension_mask("state_decoded", index)

            item["input_masks"].batch_size = [seq_len]
            item["target_masks"].batch_size = [seq_len]

        self.dispose_data(index)
        return item

    def dispose_data(self, index: int) -> None:
        ...

    @profile_range("set_state", category="wm_dataset", domain="world_machine")
    def set_state(self, indexes: torch.Tensor, states: torch.Tensor) -> None:
        if self._states is None:
            dtype = states.dtype
            states_shape = list(states.shape)
            states_shape[0] = self._size

            if self._map_state_to_disk:
                i = 0
                while self._states is None:
                    while True:
                        filename = f"TempStates_{self.__class__.__name__}_{i}.bin"

                        if not os.path.exists(filename):
                            break

                        i += 1

                    try:
                        self._states = MemoryMappedTensor.empty(
                            states_shape, dtype=dtype, filename=filename)

                    except RuntimeError as e:
                        print(e)
                        pass

                self._states_filename = filename
                WorldMachineDataset._states_filenames.append(filename)
            else:
                self._states = torch.empty(states_shape, dtype=dtype)

        self._states[indexes.cpu()] = states.detach().cpu()

    @classmethod
    def _delete_files(cls):
        for filename in cls._states_filenames:
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass


atexit.register(WorldMachineDataset._delete_files)
