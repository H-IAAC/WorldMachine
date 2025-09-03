
from hamilton.function_modifiers import source, value

from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.load_multiple_metrics import (
    load_multiple_metrics)
from world_machine_experiments.shared.save_metrics import save_metrics
from world_machine_experiments.shared.save_plots import save_plots
from world_machine_experiments.shared.statistics import consolidated_metrics
from world_machine_experiments.toy1d.base import toy1d_masks_sensorial_plots

multiple_toy1d_mask_sensorial_metrics = function_variation({"metrics_name": value(
    "mask_sensorial_metrics")}, "multiple_toy1d_mask_sensorial_metrics")(load_multiple_metrics)

multiple_toy1d_consolidated_mask_sensorial_metrics = function_variation({"metrics": source(
    "multiple_toy1d_mask_sensorial_metrics")}, "multiple_toy1d_consolidated_mask_sensorial_metrics")(consolidated_metrics)

save_multiple_toy1d_consolidated_mask_sensorial_metrics = function_variation({"metrics": source(
    "multiple_toy1d_consolidated_mask_sensorial_metrics"), "metrics_name": value("toy1d_mask_sensorial_metrics")}, "save_multiple_toy1d_consolidated_mask_sensorial_metrics")(save_metrics)

multiple_toy1d_consolidated_mask_sensorial_plots = function_variation({"toy1d_mask_sensorial_metrics": source(
    "multiple_toy1d_consolidated_mask_sensorial_metrics")}, "multiple_toy1d_consolidated_mask_sensorial_plots")(toy1d_masks_sensorial_plots)

save_multiple_toy1d_consolidated_mask_sensorial_plots = function_variation({"plots": source(
    "multiple_toy1d_consolidated_mask_sensorial_plots")}, "save_multiple_toy1d_consolidated_mask_sensorial_plots")(save_plots)
