from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from hamilton.function_modifiers import source, value
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from world_machine import WorldMachine
from world_machine.evaluate import MetricsGenerator
from world_machine.train import CriterionSet, Trainer
from world_machine.train.stages import SensorialMasker
from world_machine.train.trainer import DatasetPassMode
from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.save_metrics import save_metrics
from world_machine_experiments.shared.save_plots import save_plots
from world_machine_experiments.shared.train_plots import train_plots


def toy1d_mask_sensorial_metrics(toy1d_model_trained: WorldMachine,
                                 toy1d_dataloaders: dict[str, DataLoader],
                                 toy1d_criterion_set: CriterionSet) -> dict[str, np.ndarray]:

    metrics_generator = MetricsGenerator(toy1d_criterion_set)
    mask_sensorial_percentage = np.array([0, 0.25, 0.5, 0.75, 1.0])

    all_metrics: dict[str, dict[str, float]] = {}
    for pct in mask_sensorial_percentage:
        sm = SensorialMasker(pct, True)
        metrics_generator.stages = [sm]

        metrics = metrics_generator(toy1d_model_trained,
                                    toy1d_dataloaders["val"],
                                    compute_prediction=False,
                                    compute_use_state=False,
                                    compute_prediction_shallow=False)

        all_metrics[str(pct)] = metrics["normal"]

    criterions = list(all_metrics["0.0"].keys())
    result = {}
    result["mask_sensorial_percentage"] = mask_sensorial_percentage
    for criterion in criterions:
        metric = []

        for pct in mask_sensorial_percentage:
            metric.append(all_metrics[str(pct)][criterion])

        result[criterion] = np.array(metric)

    return result


toy1d_masks_sensorial_plots = function_variation({
    "train_history": source("toy1d_mask_sensorial_metrics"),
    "x_axis": value("mask_sensorial_percentage"),
    "series_names": value([]),
    "plot_prefix": value("mask_sensorial"),
}, "toy1d_masks_sensorial_plots")(train_plots)


save_toy1d_mask_sensorial_plot = function_variation({"plots": source(
    "toy1d_masks_sensorial_plots")}, "save_toy1d_mask_sensorial_plot")(save_plots)

save_toy1d_mask_sensorial_metrics = function_variation({"metrics": source(
    "toy1d_mask_sensorial_metrics"), "metrics_name": value("mask_sensorial_metrics")}, "save_toy1d_mask_sensorial_metrics")(save_metrics)
