import tensordict
import torch
from tensordict import TensorDict
from torch.nn import Module
from torch.optim import Optimizer

from world_machine import WorldMachine
from world_machine.data import WorldMachineDataset
from world_machine.train import DatasetPassMode

from .train_stage import TrainStage


class SequenceBreaker(TrainStage):
    def __init__(self):
        super().__init__(3)

        self.break_index: int

    def pre_segment(self, itens: list[TensorDict], losses: dict, batch_size: int,
                    seq_len: int, epoch_index: int, device: torch.device,
                    state_size: int, mode: DatasetPassMode) -> None:

        if mode == DatasetPassMode.MODE_TRAIN:
            break_index = int(self.np_generator.uniform(0, seq_len))

            if break_index == 0:
                return

            item = itens[0]

            index = item["index"]
            del item["index"]
            item.batch_size = [batch_size, seq_len]

            segments: list[TensorDict] = []

            for slice in ["start", "end"]:
                segment = {}

                if slice == "start":
                    segment = item[:, :break_index]
                else:
                    segment = item[:, break_index:]

                segment.batch_size = [batch_size]

                segment["index"] = index
                segments.append(segment)

            item.batch_size = [batch_size]
            item["index"] = index

            self.break_index = break_index

            itens.clear()
            itens.extend(segments)

    def post_segment(self, itens: list[TensorDict], losses: dict, dataset: WorldMachineDataset, epoch_index: int,
                     criterions: dict[str, dict[str, Module]], mode: DatasetPassMode,
                     device: torch.device, train_criterions: dict[str, dict[str, float]]) -> None:
        if mode == DatasetPassMode.MODE_TRAIN:
            index = itens[0]["index"]
            batch_size = itens[0].batch_size[0]

            for item in itens:
                del item["index"]
                seq_len = item["inputs"][next(
                    iter(item["inputs"].keys()))].shape[1]
                item.batch_size = [batch_size, seq_len]

            reconstructed_item: TensorDict = tensordict.cat(itens, dim=1)
            reconstructed_item.batch_size = [batch_size]
            reconstructed_item["index"] = index

            itens.clear()
            itens.append(reconstructed_item)
