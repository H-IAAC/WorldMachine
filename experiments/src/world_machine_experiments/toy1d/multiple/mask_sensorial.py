import glob
import os

import numpy as np
from hamilton.function_modifiers import source, value

from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.save_plots import save_plots
from world_machine_experiments.shared.save_train_history import (
    save_train_history)
from world_machine_experiments.shared.statistics import (
    consolidated_train_statistics)
from world_machine_experiments.toy1d.base import toy1d_masks_sensorial_plots


def multiple_toy1d_mask_sensorial_metrics(output_dir: str) -> list[dict[str, np.ndarray]]:
    paths = glob.glob(os.path.join(output_dir, "*"))

    result: list[dict[str, np.ndarray]] = []
    for path in paths:
        if os.path.isdir(path):
            history_path = os.path.join(path, "mask_sensorial_metrics.npz")

            result.append(dict(np.load(history_path)))

    return result


multiple_toy1d_consolidated_mask_sensorial_metrics = function_variation({"training_infos": source(
    "multiple_toy1d_mask_sensorial_metrics")}, "multiple_toy1d_consolidated_mask_sensorial_metrics")(consolidated_train_statistics)

save_multiple_toy1d_consolidated_mask_sensorial_metrics = function_variation({"train_history": source(
    "multiple_toy1d_consolidated_mask_sensorial_metrics"), "history_name": value("toy1d_mask_sensorial_metrics")}, "save_multiple_toy1d_consolidated_mask_sensorial_metrics")(save_train_history)

multiple_toy1d_consolidated_mask_sensorial_plots = function_variation({"toy1d_mask_sensorial_metrics": source(
    "multiple_toy1d_consolidated_mask_sensorial_metrics")}, "multiple_toy1d_consolidated_mask_sensorial_plots")(toy1d_masks_sensorial_plots)

save_multiple_toy1d_consolidated_mask_sensorial_plots = function_variation({"plots": source(
    "multiple_toy1d_consolidated_mask_sensorial_plots")}, "save_multiple_toy1d_consolidated_mask_sensorial_plots")(save_plots)
