from hamilton.function_modifiers import source, value

from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.load_multiple_metrics import (
    load_multiple_metrics)
from world_machine_experiments.shared.save_metrics import save_metrics
from world_machine_experiments.shared.statistics import consolidated_metrics

multiple_toy1d_metrics = function_variation({"metrics_name": value(
    "metrics")}, "multiple_toy1d_metrics")(load_multiple_metrics)

multiple_toy1d_consolidated_metrics = function_variation({"metrics": source(
    "multiple_toy1d_metrics")}, "multiple_toy1d_consolidated_metrics")(consolidated_metrics)

save_multiple_toy1d_consolidated_metrics = function_variation({"metrics": source(
    "multiple_toy1d_consolidated_metrics"), "metrics_name": value("toy1d_metrics")}, "save_multiple_toy1d_consolidated_metrics")(save_metrics)
