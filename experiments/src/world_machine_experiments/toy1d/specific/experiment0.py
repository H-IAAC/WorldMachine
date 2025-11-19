import os
import re
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch
from hamilton.function_modifiers import source
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tensordict import TensorDict

from world_machine_experiments.shared.acronyms import format_name
from world_machine_experiments.shared.function_variation import (
    function_variation)
from world_machine_experiments.shared.metric_plot import (
    metrics_bar_with_errorbar, metrics_boxplot)
from world_machine_experiments.shared.save_metrics import (
    load_metrics, load_multiple_metrics)
from world_machine_experiments.shared.save_plots import save_plots

variations_sorted = ["Base", "SensoryMask", "CompleteProtocol"]


def train_history(data_dir: str) -> dict[str, dict]:
    train_history_interm = load_multiple_metrics(
        data_dir, "toy1d_train_history")

    result = {}
    for name in variations_sorted:
        result[name] = train_history_interm[name]

    return result


def metrics(data_dir: str) -> dict[str, dict]:
    metrics = load_multiple_metrics(data_dir,
                                    "toy1d_metrics")
    mask_sensory_metrics = load_multiple_metrics(data_dir,
                                                 "toy1d_mask_sensory_metrics")

    for var_name in metrics:
        metrics[var_name]["means"]["mask_sensory@100"] = {}
        metrics[var_name]["stds"]["mask_sensory@100"] = {}

        for loss_name in mask_sensory_metrics[var_name]["means"]:
            if loss_name == "mask_sensory_percentage":
                continue

            metrics[var_name]["means"]["mask_sensory@100"][loss_name] = mask_sensory_metrics[var_name]["means"][loss_name][-1]
            metrics[var_name]["stds"]["mask_sensory@100"][loss_name] = mask_sensory_metrics[var_name]["stds"][loss_name][-1]

    metrics_sorted = {}
    for name in variations_sorted:
        metrics_sorted[name] = metrics[name]

    return metrics_sorted


def metrics_full(data_dir: str) -> dict[str, dict]:
    metrics_full = {}

    for name in variations_sorted:
        path = os.path.join(data_dir, name)

        metrics_full[name] = load_multiple_metrics(path, "metrics")

        mask_sensory_metrics = load_multiple_metrics(
            path, "mask_sensory_metrics")

        for run_name in metrics_full[name]:
            metrics_full[name][run_name]["mask_sensory@100"] = {}

            for loss_name in mask_sensory_metrics[run_name]:
                if loss_name == "mask_sensory_percentage":
                    continue

                metrics_full[name][run_name]["mask_sensory@100"][loss_name] = mask_sensory_metrics[run_name][loss_name][-1]

    return metrics_full


def samples(data_dir: str) -> dict[str, dict[str, TensorDict]]:
    samples = {}

    for variation in variations_sorted:
        samples[variation] = {}

        base_path = os.path.join(
            data_dir, variation, "run_4", "metrics_logits")

        for name in ["normal", "use_state", "prediction_local", "prediction", "prediction_shallow", "targets"]:
            path = os.path.join(base_path, name)
            samples[variation][name] = TensorDict.load(path)

    return samples


color_map = {
    "Base": "#60BF60",
    "SensoryMask": "#D36767",
    "CompleteProtocol": "#6060BF"
}

palette = list(color_map.values())


def train_plots(train_history: dict[str, dict]) -> dict[str, Figure]:

    n_epoch = len(train_history[next(iter(train_history))]
                  ["means"]["optimizer_loss_train"])

    epochs = range(n_epoch)

    figures = {}

    metrics = ["optimizer_loss", "state_decoded_mse", "state_decoded_0.1sdtw"]

    for i, metric_name in enumerate(metrics):
        metric_name = metrics[i]
        metric_label = format_name(metric_name)
        plot_name = f"train_history_{metric_name}"

        fig, axs = plt.subplots(1, 3, dpi=600)
        axs: list[Axes]

        for j, name in enumerate(train_history):

            means = train_history[name]["means"][f"{metric_name}_train"]
            stds = train_history[name]["stds"][f"{metric_name}_train"]

            axs[j].plot(epochs, means, "o-", label="Train",
                        color=color_map["CompleteProtocol"])
            axs[j].fill_between(epochs, means-stds, means+stds,
                                color=color_map["CompleteProtocol"], alpha=.2)

            means = train_history[name]["means"][f"{metric_name}_val"]
            stds = train_history[name]["stds"][f"{metric_name}_val"]

            axs[j].plot(epochs, means, "o-", label="Validation",
                        color=color_map["SensoryMask"])
            axs[j].fill_between(epochs, means-stds, means+stds,
                                color=color_map["SensoryMask"], alpha=.2)

            axs[j].set_xlabel("Epochs")

            plt.sca(axs[j])
            if j == 0:
                axs[j].set_ylabel(metric_label)

            plt.grid(True, "both", "y")

            if i != 2:
                axs[j].set_yscale("log")

            name = " ".join(re.sub(r"([A-Z])|(\d+)", r" \1\2", name).split())
            axs[j].set_title(name)

        maximum = -np.inf
        minimum = np.inf

        for ax in axs:
            maximum = max(maximum, ax.get_ylim()[1])
            minimum = min(minimum, ax.get_ylim()[0])

        for i, ax in enumerate(axs):
            ax.set_ylim(minimum, maximum)

            if i != 0:
                with warnings.catch_warnings(action="ignore"):
                    ax.set_yticklabels([""]*len(ax.get_yticklabels()))

        plt.suptitle(f"Model Train - {metric_label}")
        plt.legend(bbox_to_anchor=(1.375, 0.98), borderaxespad=0)

        figures[plot_name] = fig

    return figures


def metrics_box_plots(metrics_full: dict[str, dict]) -> dict[str, Figure]:

    figures = {}
    for criterion in ["mse", "0.1sdtw"]:
        fig, ax = plt.subplots(dpi=600)

        fig.suptitle("Variation Metrics")
        ax.set_title(format_name(criterion))

        metrics_boxplot(metrics_full, f"state_decoded_{criterion}",
                        "Variations", palette, ax)
        ax.set_ylabel(f"State Decoded - {format_name(criterion)}")

        ax.set_yscale("log")

        handles, labels = ax.get_legend_handles_labels()
        for i in range(len(labels)):
            labels[i] = labels[i].replace(" ", "\n")
            labels[i] = labels[i].replace("@", "\n@")
        ax.legend(handles, labels,
                  bbox_to_anchor=(0.25, 0.98), borderaxespad=0)

        xticks = plt.xticks()
        labels = xticks[1]
        for i in range(len(labels)):
            labels[i] = labels[i].get_text().replace(" ", "\n")

        xticks[0][-1] += 0.25
        plt.xticks(xticks[0], labels)

        ylim = plt.ylim()
        for i in range(len(labels)-1):
            plt.vlines(i+0.5, ylim[0], ylim[1], "black", linewidth=0.5)
        plt.ylim(ylim)

        plot_name = f"metrics_box_plot_{criterion}"
        figures[plot_name] = fig

    return figures


def metrics_bar_plots(metrics: dict[str, dict]) -> dict[str, Figure]:

    figures = {}
    for criterion in ["mse", "0.1sdtw"]:
        fig, ax = plt.subplots(dpi=300, figsize=(11, 8))

        fig.suptitle("Variation Metrics")
        ax.set_title(format_name(criterion))

        metrics_bar_with_errorbar(
            metrics, f"state_decoded_{criterion}", "Variations", palette, "1 σ", ax)
        ax.set_ylabel(f"State Decoded - {format_name(criterion)}")
        ax.set_yscale("log")

        ax.get_legend().remove()
        ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

        plot_name = f"metrics_bar_plot_{criterion}"
        figures[plot_name] = fig

    return figures


def sample_plots(samples: dict[str, dict[str, TensorDict]]) -> dict[str, Figure]:
    time = np.linspace(0, 199, 200, dtype=int)

    linewidth = 0.75

    indexes = [0, 1, 2, 9, 4, 8, 10, 11, 12]

    variation_names = list(samples.keys())

    fig, axs = plt.subplots(3, 5, dpi=600, figsize=(8, 4))
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=0.05, hspace=0.05)

    index = 0
    for j in range(3):
        for i in range(5):
            row = index // 5
            column = index % 5

            axs[row, column].plot(samples[variation_names[j]]["targets"]["state_decoded"][indexes[i]],
                                  label="Target", color="black", linewidth=linewidth)

            axs[row, column].plot(samples[variation_names[j]]["normal"]["state_decoded"][indexes[i]],
                                  label="Normal", color="tab:blue", linewidth=linewidth)

            axs[row, column].plot(time,
                                  samples[variation_names[j]
                                          ]["prediction_local"]["state_decoded"][indexes[i]],
                                  label="Prediction Local", color="tab:purple", linewidth=linewidth)

            axs[row, column].plot(time[:100],
                                  samples[variation_names[j]
                                          ]["use_state"]["state_decoded"][indexes[i]],
                                  label="Use State",  color="tab:orange", linewidth=linewidth)

            axs[row, column].plot(time[100:],
                                  samples[variation_names[j]
                                          ]["prediction"]["state_decoded"][indexes[i]],
                                  label="Prediction", color="tab:green", linewidth=linewidth)

            axs[row, column].plot(time[100:],
                                  samples[variation_names[j]
                                          ]["prediction_shallow"]["state_decoded"][indexes[i]],
                                  label="Prediction Shallow", color="tab:red", linewidth=linewidth)

            axs[row, column].set_xticks([])
            axs[row, column].set_yticks([])

            axs[row, column].axvline(100, color="black")

            index += 1

    for i, name in enumerate(variation_names):
        name = format_name(name).replace(" ", "\n")
        axs[i, 0].set_ylabel(name)
        # axs[i, 4].set_ylabel(name)

    # for i in range(3):
    #    axs[i, 4].yaxis.set_label_position("right")
    #    axs[i, 4].yaxis.tick_right()

    for i in range(5):
        # axs[0, i].set_title(str(i))
        axs[2, i].set_xlabel(str(i))

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    for i in range(len(labels)):
        labels[i] = labels[i].replace(" ", "\n")
        labels[i] = labels[i].replace("@", "\n@")
    ax.legend(handles, labels,
              bbox_to_anchor=(2.1, 3.1), loc='upper right')

    # plt.legend(bbox_to_anchor=(1.9, 3.1), loc='upper right')

    plt.suptitle("Inference Samples")

    return {"state_decoded_samples": fig}


def state_analysis_plots(samples: dict[str, dict[str, TensorDict]]) -> dict[str, Figure]:
    variation_names = list(samples.keys())

    indexes = [0, 1, 2, 9, 4, 8, 10, 11, 12]

    figures = {}
    for name in variation_names:
        fig, axs = plt.subplots(3, 5, dpi=300, figsize=(16, 8))
        axs: list[list[Axes]]
        plt.subplots_adjust(left=None, bottom=None, right=None,
                            top=None, wspace=0.05, hspace=0.05)
        index = 0

        for i in range(5):
            row = index // 5
            column = index % 5

            axs[row, column].plot(samples[name]["targets"]
                                  ["state_decoded"][indexes[i]], label="Target")

            if column not in [0, 4]:
                axs[row, column].set_yticks([])
            axs[row, column].set_xticks([])

            index += 1

        for i in range(5):
            row = index // 5
            column = index % 5

            axs[row, column].plot(samples[name]
                                  ["normal"]["state"][indexes[i]])

            if column not in [0, 4]:
                axs[row, column].set_yticks([])
            axs[row, column].set_xticks([])

            index += 1

        for i in range(5):
            row = index // 5
            column = index % 5

            ws_09_count = (torch.abs(
                samples[name]["normal"]["state"][indexes[i]]) > 0.9).sum(axis=1)
            ws_09_count = ws_09_count / \
                samples[name
                        ]["normal"]["state"][indexes[i]].shape[1]

            axs[row, column].plot(ws_09_count, alpha=0.7)

            if column not in [0, 4]:
                axs[row, column].set_yticks([])
            axs[row, column].set_xticks([])

            axs[row, column].set_ylim(0, 1)

            index += 1

        index = 0

        for i in range(5):
            for _ in range(3):
                row = index // 5
                column = index % 5

                axs[row, column].axvline(50, color="black")
                axs[row, column].axvline(100, color="black")
                axs[row, column].axvline(150, color="black")

                index += 1

        row_names = ["Target", "State", "Count(|State|>0.9)"]
        for i, r_name in enumerate(row_names):
            axs[i, 0].set_ylabel(r_name)
            axs[i, 4].set_ylabel(r_name)

        for i in range(3):
            axs[i, 4].yaxis.set_label_position("right")
            axs[i, 4].yaxis.tick_right()

        for i in range(5):
            axs[0, i].set_title(str(i))
            axs[2, i].set_xlabel(str(i))

        plt.suptitle(f"State Analysis - {format_name(name)}")

        figures[f"state_analysis_{name}"] = fig

    return figures


def target_state09_correlation_plot(samples: dict[str, dict[str, TensorDict]]) -> dict[str, Figure]:
    variation_names = list(samples.keys())

    fig, axs = plt.subplots(1, 3, dpi=300, figsize=(15, 4))
    axs: list[Axes]

    for i, name in enumerate(variation_names):
        ws_09_count = (
            torch.abs(samples[name]["normal"]["state"]) > 0.9).sum(axis=2)
        target = samples[name]["targets"]["state_decoded"][:, :, 0]
        sns.histplot(stats.pearsonr(
            ws_09_count, target, axis=1).statistic, ax=axs[i])

        axs[i].set_title(format_name(name))
        axs[i].set_xlim(-1, 1)
        axs[i].set_xlabel("Pearson Coefficient")

    maximum = -np.inf
    for ax in axs:
        maximum = max(maximum, ax.get_ylim()[1])

    for ax in axs:
        ax.set_ylim(0, maximum)

    plt.suptitle("Correlation Between Target and Count(|State|>0.9)")
    plt.tight_layout()

    figures = {"target_state09_correlation": fig}
    return figures


save_train_plots = function_variation({"plots": source(
    "train_plots")}, "save_train_plots")(save_plots)

save_metrics_box_plots = function_variation({"plots": source(
    "metrics_box_plots")}, "save_metrics_box_plots")(save_plots)

save_metrics_bar_plots = function_variation({"plots": source(
    "metrics_bar_plots")}, "save_metrics_bar_plots")(save_plots)

save_samples_plots = function_variation({"plots": source(
    "sample_plots")}, "save_samples_plots")(save_plots)

save_state_analysis_plots = function_variation({"plots": source(
    "state_analysis_plots")}, "save_state_analysis_plots")(save_plots)

save_target_state09_correlation_plot = function_variation({"plots": source(
    "target_state09_correlation_plot")}, "save_target_state09_correlation_plot")(save_plots)
