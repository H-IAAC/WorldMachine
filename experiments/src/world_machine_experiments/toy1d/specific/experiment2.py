import os

import matplotlib.pyplot as plt
import numpy as np
from hamilton.function_modifiers import source, value
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tensordict import TensorDict

from world_machine_experiments.shared.function_variation import (
    function_variation)
from world_machine_experiments.shared.metric_plot import (
    metrics_bar_with_errorbar, metrics_boxplot)
from world_machine_experiments.shared.save_metrics import (
    load_metrics, load_multiple_metrics)
from world_machine_experiments.shared.save_plots import save_plots


def samples(data_dir: str) -> dict[str, TensorDict]:
    samples = {}

    logits_path = os.path.join(
        data_dir, "CosineAnnealingWithWarmup", "run_0", "metrics_logits")

    samples = {}
    for name in ["normal", "use_state", "prediction_local", "prediction", "prediction_shallow", "targets"]:
        path = os.path.join(logits_path, name)
        samples[name] = TensorDict.load(path)

    return samples


def toy1d_samples_plots(samples: dict[str, TensorDict]) -> dict[str, Figure]:
    fig, axs = plt.subplots(3, 2, dpi=900)

    axs: list[list[Axes]]

    axs_flat: list[Axes] = axs.flatten()

    for i in range(3):
        axs[i][0].plot(samples["targets"]["state_decoded"]
                       [i, :, 0], color="black")

    for i in range(3):
        axs[i][1].plot(samples["targets"]["measurement"][i])

    for ax in axs_flat:
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(0, 200)
        ax.set_yticks(np.arange(-1, 1+0.5, 0.5))
        ax.grid(True, "both", "y")

        ax.vlines(100, -1.1, 1.1, color="gray", linewidth=0.8)

    for i in range(3):
        axs[i][1].set_yticklabels([])
        axs[i][1].tick_params(axis='y', length=0)

    for i in range(2):
        axs[i][0].set_xticks([])
        axs[i][1].set_xticks([])

    axs[0][0].set_title("External State")
    axs[0][1].set_title("Measurement")

    fig.subplots_adjust(wspace=0.08, hspace=0.05)

    plt.suptitle("Toy1D Dataset Samples")

    return {"toy1d_samples": fig}


save_toy1d_samples_plots = function_variation({"plots": source(
    "toy1d_samples_plots")}, "save_toy1d_samples_plots")(save_plots)


def prediction_shallow_samples_plots(samples: dict[str, TensorDict]) -> dict[str, Figure]:
    fig, axs = plt.subplots(3, 1, figsize=[6.4/2, 4.8], dpi=900)
    axs: list[Axes]

    for i in range(3):
        plt.sca(axs[i])

        plt.plot(samples["targets"]["state_decoded"][i, :, 0],
                 "--", color="black", label="External State")
        plt.plot(np.arange(100, 200), samples["prediction_shallow"]
                 ["state_decoded"][i, :, 0], color="red", label="Prediction Shallow")
        plt.xlim(0, 200)
        plt.ylim(-1.1, 1.1)

        plt.vlines(100, -1.1, 1.1, color="gray", linewidth=0.80)

    plt.suptitle("Prediction Shallow Samples")

    axs[0].legend(bbox_to_anchor=(0.8, -2.75), borderaxespad=0)

    return {"prediction_shallow_samples": fig}


save_prediction_shallow_samples_plots = function_variation({"plots": source(
    "prediction_shallow_samples_plots")}, "save_prediction_shallow_samples_plots")(save_plots)


metrics = function_variation({"output_dir": source("data_dir"),
                              "metrics_name": value("toy1d_metrics")},
                             "metrics")(load_multiple_metrics)


def variation_performance_plots(metrics: dict[str, dict]) -> dict[str, Figure]:
    fig = plt.figure(dpi=900)
    ax = plt.gca()
    metrics_bar_with_errorbar(
        metrics, "state_decoded_mse", "Variations", errorbar=False, ax=ax)

    xlabels = plt.xticks()[1]
    for i in range(len(xlabels)):
        xlabels[i] = xlabels[i].get_text().replace(" ", "\n")
    plt.xticks(plt.xticks()[0], xlabels)

    plt.xlabel("Task")
    plt.ylabel("MSE")

    plt.title("Variation Performance Across Tasks")

    plt.yscale("log")

    return {"variation_performance": fig}


save_variation_performance_plots = function_variation({"plots": source(
    "variation_performance_plots")}, "save_variation_performance_plots")(save_plots)
