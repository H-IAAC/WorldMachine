from hamilton.function_modifiers import source, value

from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.load_train_history import (
    load_train_history)
from world_machine_experiments.shared.parameter_variation_plots import (
    parameter_variation_plots)
from world_machine_experiments.shared.save_plots import save_plots

toy1d_load_mask_sensorial_metrics = function_variation({"history_file_name": value(
    "toy1d_mask_sensorial_metrics")}, "toy1d_load_mask_sensorial_metrics")(load_train_history)

toy1d_parameter_variation_mask_sensorial_plots = function_variation({"train_history": source(
    "toy1d_load_mask_sensorial_metrics"), "x_axis": value("mask_sensorial_percentage")}, "toy1d_parameter_variation_mask_sensorial_plots")(parameter_variation_plots)

save_toy1d_parameter_variation_mask_sensorial_plots = function_variation({"plots": source(
    "toy1d_parameter_variation_mask_sensorial_plots")}, "save_toy1d_parameter_variation_mask_sensorial_plots")(save_plots)
