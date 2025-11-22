import numba
import numpy as np
import torch
from tensordict import TensorDict
from torch.nn import Module
from torch.optim import Optimizer

from world_machine.profile import profile_range
from world_machine.train.mode import DatasetPassMode
from world_machine.train.scheduler import ConstantScheduler, ParameterScheduler
from world_machine.world_machine import WorldMachine

from .train_stage import TrainStage


class SensoryMasker(TrainStage):
    def __init__(self, mask_percentage: float | dict[str, float | ParameterScheduler] | ParameterScheduler,
                 force_sensory_mask: bool = False):
        super().__init__(3)

        self._force_sensory_mask = force_sensory_mask

        if isinstance(mask_percentage, float):
            mask_percentage = ConstantScheduler(mask_percentage, 0)
        elif isinstance(mask_percentage, dict):
            for name in mask_percentage:
                if isinstance(mask_percentage[name], float):
                    mask_percentage[name] = ConstantScheduler(
                        mask_percentage[name], 0)

        self._mask_percentage: ParameterScheduler | dict[str,
                                                         ParameterScheduler] = mask_percentage

    @profile_range("SensoryMasker_generate_mask_percentage", category="train_stage", domain="world_machine")
    def _generate_mask_percentage(self, sensory_channels: list[str], epoch_index: int) -> dict[str, float]:
        mask_sensory_data = self._mask_percentage

        if isinstance(mask_sensory_data, ParameterScheduler):
            mask_percentage = mask_sensory_data(epoch_index)
            mask_percentage = {
                channel: mask_percentage for channel in sensory_channels}
        else:
            mask_percentage = mask_sensory_data.copy()

            for channel in mask_percentage:
                mask_percentage[channel] = mask_percentage[channel](
                    epoch_index)

        mask_percentage = {channel: float(
            mask_percentage[channel]) for channel in mask_percentage}

        return mask_percentage

    def pre_segment(self, itens: list[TensorDict], losses: dict, batch_size: int,
                    seq_len: int, epoch_index: int, device: torch.device,
                    state_size: int, mode: DatasetPassMode, model: WorldMachine) -> None:
        if mode == DatasetPassMode.MODE_TRAIN or self._force_sensory_mask:
            item = itens[0]

            inputs: TensorDict = item["inputs"]

            with torch.no_grad():
                if "input_masks" not in item:
                    sensory_masks = TensorDict(
                        device=device, batch_size=batch_size)

                    sensory_data: TensorDict = inputs
                    for name in sensory_data.keys():
                        sensory_masks[name] = torch.ones(
                            (batch_size, seq_len), dtype=bool, device=device)
                else:
                    sensory_masks = item["input_masks"]

                mask_percentage = self._generate_mask_percentage(
                    sensory_masks.keys(), epoch_index)

                sensory_masks = generate_masks(sensory_masks,
                                               mask_percentage, batch_size, device)

            item["input_masks"] = sensory_masks
            item["input_masks"].batch_size = [batch_size, seq_len]


@profile_range("SensoryMasker_mask_mask", category="train_stage", domain="world_machine")
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


@profile_range("SensoryMasker_generate_masks", category="train_stage", domain="world_machine")
def generate_masks(sensory_masks: TensorDict, mask_percentage: dict[str, float], batch_size: int, device):

    for sensory_channel in sensory_masks.keys():
        if sensory_channel in mask_percentage:
            channel_percentage = mask_percentage[sensory_channel]

            masks = sensory_masks[sensory_channel].cpu().numpy()
            sensory_masks[sensory_channel] = torch.tensor(
                mask_mask(masks, channel_percentage, batch_size), device=device)

    return sensory_masks
