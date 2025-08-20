from torch.nn import MSELoss

from world_machine.train import CriterionSet, Trainer
from world_machine.train.stages import (
    LossManager, SensorialMasker, SequenceBreaker, ShortTimeRecaller,
    StateManager)


def get_trainer() -> Trainer:
    cs = CriterionSet()

    cs.add_sensorial_criterion("mse", "dim0", MSELoss(), True)
    cs.add_sensorial_criterion("mse", "dim1", MSELoss(), True)

    stages = []
    stages.append(SensorialMasker(0.5))
    stages.append(StateManager(1, True))
    stages.append(SequenceBreaker(2, True))

    dimension_sizes = {}
    criterions = {}

    for dim in ["dim0", "dim1"]:
        dimension_sizes[dim] = 3
        criterions[dim] = MSELoss()

    stages.append(ShortTimeRecaller(dimension_sizes=dimension_sizes,
                                    criterions=criterions,
                                    n_past=2,
                                    n_future=2,
                                    stride_past=2,
                                    stride_future=2))

    stages.append(LossManager())

    trainer = Trainer(cs, stages, 0)

    return trainer
