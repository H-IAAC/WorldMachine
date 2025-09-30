from world_machine.layers import MultiHeadAttention

from .train_stage import TrainStage


class LocalSetter(TrainStage):
    def __init__(self, local_chance: float = 0.5):
        super().__init__(3)

        self._local_chance = local_chance

    def pre_segment(self, itens, losses, batch_size, seq_len, epoch_index, device, state_size, mode, model):

        local = self.np_generator.random() <= self._local_chance
        for module in model.modules():
            if isinstance(module, MultiHeadAttention):
                module.local_only = local

    def post_train(self, model, criterions, train_criterions):
        for module in model.modules():
            if isinstance(module, MultiHeadAttention):
                module.local_only = False
