
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler

from world_machine.train.mode import DatasetPassMode
from world_machine.world_machine import WorldMachine

from .train_stage import TrainStage


class LRScheduler_Step(TrainStage):
    def __init__(self, scheduler: LRScheduler):
        super().__init__(0)
        self._scheduler = scheduler

    def post_batch(self,
                   model: WorldMachine,
                   losses: dict,
                   criterions: dict[str, dict[str, Module]],
                   train_criterions: dict[str, dict[str, float]],
                   mode: DatasetPassMode) -> None:

        if mode == DatasetPassMode.MODE_TRAIN:
            self._scheduler.step()
