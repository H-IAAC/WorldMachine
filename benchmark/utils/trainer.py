from torch.nn import MSELoss

from world_machine.train import CriterionSet, Trainer
from world_machine.train.stages import (
    LossManager, SensoryMasker, SequenceBreaker, ShortTimeRecaller,
    StateManager)


def get_trainer() -> Trainer:
    cs = CriterionSet()

    cs.add_sensory_criterion("mse", "channel0", MSELoss(), True)
    cs.add_sensory_criterion("mse", "channel1", MSELoss(), True)

    stages = []
    stages.append(SensoryMasker(0.5))
    stages.append(StateManager(1, True))
    stages.append(SequenceBreaker(2, True))

    channel_sizes = {}
    criterions = {}

    for channel in ["channel0", "channel1"]:
        channel_sizes[channel] = 3
        criterions[channel] = MSELoss()

    stages.append(ShortTimeRecaller(channel_sizes=channel_sizes,
                                    criterions=criterions,
                                    n_past=2,
                                    n_future=2,
                                    stride_past=2,
                                    stride_future=2))

    stages.append(LossManager())

    trainer = Trainer(cs, stages, 0)

    return trainer
