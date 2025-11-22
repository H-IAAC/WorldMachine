from torch.nn import Module
from torch.optim import Optimizer

from world_machine.world_machine import WorldMachine

from .train_stage import TrainStage


class LocalSetter(TrainStage):
    def __init__(self, local_chance: float = 0.5):
        super().__init__(3)

        self._local_chance = local_chance

    def pre_segment(self, itens, losses, batch_size, seq_len, epoch_index, device, state_size, mode, model):
        local = self.np_generator.random() <= self._local_chance
        model.local_mode = local

    def post_train(self, model: WorldMachine,
                   criterions: dict[str, dict[str, Module]],
                   train_criterions: dict[str, dict[str, float]],
                   optimizer: Optimizer) -> None:
        model.local_mode = False
