import numba
import numpy as np
import torch
from tensordict import TensorDict
from torch.nn import Module
from torch.optim import Optimizer

from world_machine import WorldMachine
from world_machine.train import DatasetPassMode
from world_machine.train.scheduler import ConstantScheduler, ParameterScheduler

from .train_stage import TrainStage


class SensorialMasker(TrainStage):
    def __init__(self, mask_percentage: float | dict[str, float | ParameterScheduler] | ParameterScheduler,
                 force_sensorial_mask: bool = False):
        super().__init__(4)

        self._force_sensorial_mask = force_sensorial_mask

        if isinstance(mask_percentage, float):
            mask_percentage = ConstantScheduler(mask_percentage, 0)
        elif isinstance(mask_percentage, dict):
            for name in mask_percentage:
                if isinstance(mask_percentage[name], float):
                    mask_percentage[name] = ConstantScheduler(
                        mask_percentage[name], 0)

        self._mask_percentage: ParameterScheduler | dict[str,
                                                         ParameterScheduler] = mask_percentage

    def _generate_mask_percentage(self, sensorial_dimensions: list[str], epoch_index: int) -> dict[str, float]:
        mask_sensorial_data = self._mask_percentage

        if isinstance(mask_sensorial_data, ParameterScheduler):
            mask_percentage = mask_sensorial_data(epoch_index)
            mask_percentage = {
                dim: mask_percentage for dim in sensorial_dimensions}
        else:
            mask_percentage = mask_sensorial_data.copy()

            for dim in mask_percentage:
                mask_percentage[dim] = mask_percentage[dim](epoch_index)

        mask_percentage = {dim: float(
            mask_percentage[dim]) for dim in mask_percentage}

        return mask_percentage

    def pre_forward(self, item_index: int,  itens: list[TensorDict], mode: DatasetPassMode, batch_size: int, device: torch.device, epoch_index: int) -> None:

        if mode == DatasetPassMode.MODE_TRAIN or self._force_sensorial_mask:
            item = itens[item_index]

            inputs: TensorDict = item["inputs"]
            seq_len = item["inputs"][next(
                iter(item["inputs"].keys()))].shape[1]

            with torch.no_grad():
                if "input_masks" not in item:
                    sensorial_masks = TensorDict(
                        device=device, batch_size=batch_size)

                    sensorial_data: TensorDict = inputs
                    for name in sensorial_data.keys():
                        sensorial_masks[name] = torch.ones(
                            (batch_size, seq_len), dtype=bool, device=device)
                else:
                    sensorial_masks = item["input_masks"]

                mask_percentage = self._generate_mask_percentage(
                    sensorial_masks.keys(), epoch_index)

                sensorial_masks = generate_masks(sensorial_masks,
                                                 mask_percentage, batch_size, device)

            item["input_masks"] = sensorial_masks
            item["input_masks"].batch_size = [batch_size, seq_len]


@numba.njit(cache=True)
def mask_mask(masks: np.ndarray, mask_percentage: float, batch_size: int):
    for batch_idx in range(batch_size):
        mask: np.ndarray = masks[batch_idx]

        masked_count = (
            mask.shape[0] - mask.sum())/mask.shape[0]

        if masked_count < mask_percentage:
            idxs = np.argwhere(mask != 0).flatten()

            to_mask_count = (mask.shape[0] *
                             (mask_percentage-masked_count))
            to_mask_count = int(
                np.ceil(to_mask_count))

            to_mask = np.random.choice(  # NOSONAR
                idxs, to_mask_count, replace=False)

            masks[batch_idx][to_mask] = 0

    return masks


def generate_masks(sensorial_masks: TensorDict, mask_percentage: dict[str, float], batch_size: int, device):

    for sensorial_dim in sensorial_masks.keys():
        if sensorial_dim in mask_percentage:
            dim_percentage = mask_percentage[sensorial_dim]

            masks = sensorial_masks[sensorial_dim].cpu().numpy()
            sensorial_masks[sensorial_dim] = torch.tensor(
                mask_mask(masks, dim_percentage, batch_size), device=device)

    return sensorial_masks
