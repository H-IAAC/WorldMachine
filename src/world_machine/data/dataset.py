import abc
import atexit
import os
from collections import deque

import torch
from tensordict import MemoryMappedTensor, TensorDict
from torch.utils.data import Dataset

from world_machine.current_version import CURRENT_COMPATIBILITY_VERSION
from world_machine.profile import profile_range
from world_machine.version_upgrader import upgrade


class WorldMachineDataset(Dataset, abc.ABC):

    _states_filenames = deque()

    def __new__(cls, *args, **kwargs):
        if not getattr(cls, "_profile_wrapped", False):
            name = cls.__name__

            method_names = ["load_data", "get_channel_item",
                            "get_channel_mask", "dispose_data"]
            for method_name in method_names:
                if method_name in cls.__dict__:
                    method = getattr(cls, method_name)
                    wrapped = profile_range(
                        f"{name}_{method_name}", category="wm_dataset", domain="world_machine")(method)
                    setattr(cls, method_name, wrapped)

            cls._profile_wrapped = True

        return super().__new__(cls)

    def __init__(self, sensory_channels: list[str], size: int,
                 has_state_decoded: bool = False,
                 has_masks: bool = False,
                 map_state_to_disk: bool = True):
        super().__init__()

        self._compatibility_version = CURRENT_COMPATIBILITY_VERSION

        self._sensory_channels = sensory_channels
        self._size = size
        self._has_state_decoded = has_state_decoded
        self._has_masks = has_masks

        self._map_state_to_disk = map_state_to_disk
        self._states = None
        self._states_filename = None

        self._persist_states = False

    @property
    def persist_states(self) -> bool:
        return self._persist_states

    @persist_states.setter
    def persist_states(self, value: bool):
        if self._states_filename is not None:
            if value and self._states_filename in WorldMachineDataset._states_filenames:
                WorldMachineDataset._states_filenames.remove(
                    self._states_filename)
            elif not value and self._states_filename not in WorldMachineDataset._states_filenames:
                WorldMachineDataset._states_filenames.append(
                    self._states_filename)

        self._persist_states = value

    def __len__(self) -> int:
        return self._size

    def load_data(self, index: int) -> None:
        ...

    @abc.abstractmethod
    def get_channel_item(self, channel: str, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def get_channel_mask(self, channel: str, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @profile_range("__getitem__", category="wm_dataset", domain="world_machine")
    def __getitem__(self, index):
        item = TensorDict(
            {"inputs": TensorDict(), "targets": TensorDict(), "index": index}, batch_size=[])

        self.load_data(index)
        for channel in self._sensory_channels:
            item["inputs"][channel], item["targets"][channel] = self.get_channel_item(
                channel, index)

        if self._has_state_decoded:
            item["inputs"]["state_decoded"], item["targets"]["state_decoded"] = self.get_channel_item(
                "state_decoded", index)

        if self._states != None:
            item["inputs"]["state"] = self._states[index]

        seq_len = item["inputs"][next(iter(item["inputs"].keys()))].shape[0]

        item["inputs"].batch_size = [seq_len]
        item["targets"].batch_size = [seq_len]

        if self._has_masks:
            item["input_masks"] = TensorDict()
            item["target_masks"] = TensorDict()

            for channel in self._sensory_channels:
                (item["input_masks"][channel],
                 item["target_masks"][channel]) = self.get_channel_mask(channel, index)

            if self._has_state_decoded:
                (item["input_masks"]["state_decoded"],
                 item["target_masks"]["state_decoded"]) = self.get_channel_mask("state_decoded", index)

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

    def clear_states(self) -> None:
        del self._states
        self._states = None

        if self._states_filename is not None:
            self.delete_file(self._states_filename)

    def __del__(self) -> None:
        if self._states_filename is not None:
            self.delete_file(self._states_filename)

    @classmethod
    def delete_file(cls, filename: str) -> None:
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass

    @classmethod
    def _delete_files(cls) -> None:
        for filename in cls._states_filenames:
            cls.delete_file(filename)

    def __setstate__(self, state):
        upgrade(state)
        self.__dict__.update(state)


atexit.register(WorldMachineDataset._delete_files)
