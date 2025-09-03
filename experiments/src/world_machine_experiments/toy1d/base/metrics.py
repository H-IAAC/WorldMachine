import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from hamilton.function_modifiers import datasaver, source, value
from matplotlib.figure import Figure
from tensordict import TensorDict
from torch.utils.data import DataLoader, random_split

from world_machine import WorldMachine
from world_machine.data import WorldMachineDataLoader
from world_machine.evaluate import MetricsGenerator
from world_machine.train import CriterionSet, Trainer
from world_machine.train.trainer import DatasetPassMode
from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.save_metrics import save_metrics
from world_machine_experiments.shared.save_plots import save_plots
from world_machine_experiments.shared.save_train_history import (
    save_train_history)


def toy1d_metrics(toy1d_model_trained: WorldMachine,
                  toy1d_dataloaders: dict[str, DataLoader],
                  toy1d_criterion_set: CriterionSet) -> dict[str, dict[str, float]]:

    mg = MetricsGenerator(toy1d_criterion_set)

    metrics = mg(toy1d_model_trained, toy1d_dataloaders["val"])

    return metrics


save_toy1d_metrics = function_variation({"metrics": source(
    "toy1d_metrics"), "metrics_name": value("metrics")}, "save_toy1d_metrics")(save_metrics)


def toy1d_metrics_sample_logits(toy1d_model_trained: WorldMachine,
                                toy1d_dataloaders: dict[str, DataLoader],
                                toy1d_criterion_set: CriterionSet) -> dict[str, TensorDict]:

    dataset = toy1d_dataloaders["val"].dataset

    if len(dataset) > 32:
        dataset, _ = random_split(dataset, [32, len(dataset)-32])

    dataloader = WorldMachineDataLoader(
        dataset, batch_size=32, shuffle=False)

    mg = MetricsGenerator(toy1d_criterion_set)

    _, logits = mg(toy1d_model_trained, dataloader, return_logits=True)

    logits["targets"] = []

    for item in dataloader:
        logits["targets"].append(item["targets"])

    logits["targets"] = torch.cat(logits["targets"], 0)

    return logits


@datasaver()
def save_toy1d_metrics_sample_logits(toy1d_metrics_sample_logits: dict[str, TensorDict], output_dir: str) -> dict:
    main_path = os.path.join(output_dir, "metrics_logits")

    paths = []
    for name in toy1d_metrics_sample_logits:
        path = os.path.join(main_path, name)

        toy1d_metrics_sample_logits[name].save(path)

        paths.append(path)

    info = {"paths": paths}

    return info


def toy1d_metrics_sample_plots(toy1d_metrics_sample_logits: dict[str, TensorDict]) -> dict[str, Figure]:
    time = np.linspace(0, 199, 200, dtype=int)

    batch_size = min(toy1d_metrics_sample_logits["normal"].batch_size[0], 32)

    figures = {}
    for name in toy1d_metrics_sample_logits["normal"].keys():
        fig, axs = plt.subplots(4, 8, dpi=300, figsize=(16, 8))
        plt.subplots_adjust(left=None, bottom=None, right=None,
                            top=None, wspace=0.05, hspace=0.05)

        for i in range(batch_size):
            row = i // 8
            column = i % 8

            axs[row, column].plot(
                toy1d_metrics_sample_logits["normal"][name][i, :, 0], label="Normal", alpha=0.5)

            axs[row, column].plot(time[:100], toy1d_metrics_sample_logits["use_state"]
                                  [name][i, :, 0], label="Use State")

            axs[row, column].plot(time[100:], toy1d_metrics_sample_logits["prediction"]
                                  [name][i, :, 0], label="Prediction")

            axs[row, column].plot(time[100:], toy1d_metrics_sample_logits["prediction_shallow"]
                                  [name][i, :, 0], label="Prediction Shallow")

            if name in toy1d_metrics_sample_logits["targets"].keys():
                axs[row, column].plot(toy1d_metrics_sample_logits["targets"]
                                      [name][i, :, 0], label="Target", color="black")

            axs[row, column].set_xticks([])
            axs[row, column].set_yticks([])

            axs[row, column].axvline(100, color="black")

        plt.legend(bbox_to_anchor=(2.5, 4.5), loc='upper right')

        plt.suptitle("Metrics Inference Samples")
        plt.title(name)

        figures["metrics_sample_"+name] = fig

    return figures


save_toy1d_metrics_sample_plots = function_variation({"plots": source(
    "toy1d_metrics_sample_plots")}, "save_toy1d_metrics_sample_plots")(save_plots)
