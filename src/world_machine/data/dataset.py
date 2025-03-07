import abc

import torch
from tensordict import TensorDict
from torch.utils.data import Dataset


class WorldMachineDataset(Dataset, abc.ABC):

    def __init__(self, sensorial_dimensions: list[str], size: int,
                 has_state_decoded: bool = False,
                 has_masks: bool = False):
        super().__init__()

        self._sensorial_dimensions = sensorial_dimensions
        self._size = size
        self._has_state_decoded = has_state_decoded
        self._has_masks = has_masks

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
            {"inputs": TensorDict(), "targets": TensorDict()}, batch_size=[])

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

        self.dispose_data(index)

        return item

    def dispose_data(self, index: int) -> None:
        ...
