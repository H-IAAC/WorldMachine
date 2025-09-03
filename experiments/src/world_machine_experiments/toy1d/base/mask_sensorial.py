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


def toy1d_mask_sensorial_metrics(toy1d_model_trained: WorldMachine,
                                 toy1d_dataloaders: dict[str, DataLoader],
                                 toy1d_criterion_set: CriterionSet,
                                 discover_state: bool = False) -> dict[str, dict[str, float]]:

    metrics_generator = MetricsGenerator(toy1d_criterion_set)
    mask_sensorial_percentage = np.array([0, 0.25, 0.5, 0.75, 1.0])

    all_metrics = {}
    for pct in mask_sensorial_percentage:
        sm = SensorialMasker(pct, True)
        metrics_generator.stages = [sm]

        metrics = metrics_generator(toy1d_model_trained,
                                    toy1d_dataloaders["val"],
                                    compute_prediction=False,
                                    compute_use_state=False,
                                    compute_prediction_shallow=False)

        all_metrics[str(pct)] = metrics["normal"]

    result = defaultdict(dict)
    for key in all_metrics:
        for subkey in all_metrics[key]:
            result[subkey][key] = all_metrics[key][subkey]

    result = dict(result)  # [criterion][pct]
    return result


def toy1d_masks_sensorial_plots(toy1d_mask_sensorial_metrics: dict[str, dict[str, float]] | dict[str, dict], y_scale: str = "log") -> dict[str, Figure]:
    stds = None
    if "means" in toy1d_mask_sensorial_metrics:
        stds = toy1d_mask_sensorial_metrics["stds"]
        toy1d_mask_sensorial_metrics = toy1d_mask_sensorial_metrics["means"]

    figures = {}
    for name in toy1d_mask_sensorial_metrics:

        fig = plt.figure(dpi=300)

        toy1d_mask_sensorial_metrics[name].items
        percentages = np.array(list(toy1d_mask_sensorial_metrics[name].keys()))
        values = np.array(list(toy1d_mask_sensorial_metrics[name].values()))

        indexes = np.argsort(percentages)

        percentages = percentages[indexes]
        values = values[indexes]

        if stds is None:
            plt.plot(
                percentages, values, "o-")
        else:
            std_percentages = np.array(list(
                stds[name].keys()))
            std_values = np.array(list(stds[name].values()))

            std_indexes = np.argsort(std_percentages)
            std_percentages = std_percentages[std_indexes]
            std_values = std_values[std_indexes]

            plot_args = {"fmt": "o-", "capsize": 5.0, "markersize": 4}

            plt.errorbar(percentages,
                         values,
                         std_values, **plot_args)

        plt.suptitle("Mask Sensorial Loss")
        plt.title(name)
        plt.xlabel("Masking Percentage")
        plt.ylabel("Loss")
        # plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

        plt.yscale(y_scale)

        plt.close()

        figures[f"mask_sensorial_{name}"] = fig

    return figures


save_toy1d_mask_sensorial_plot = function_variation({"plots": source(
    "toy1d_masks_sensorial_plots")}, "save_toy1d_mask_sensorial_plot")(save_plots)

save_toy1d_mask_sensorial_metrics = function_variation({"metrics": source(
    "toy1d_mask_sensorial_metrics"), "metrics_name": value("mask_sensorial_metrics")}, "save_toy1d_mask_sensorial_metrics")(save_metrics)
