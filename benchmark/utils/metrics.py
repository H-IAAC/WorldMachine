from torch.nn import MSELoss

from world_machine.evaluate import MetricsGenerator
from world_machine.train import CriterionSet


def get_metrics_generator() -> MetricsGenerator:
    cs = CriterionSet()

    cs.add_sensory_criterion("mse", "channel0", MSELoss(), True)
    cs.add_sensory_criterion("mse", "channel1", MSELoss(), True)

    mg = MetricsGenerator(cs)

    return mg
