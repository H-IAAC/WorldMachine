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
from world_machine.train import CriterionSet
from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.acronyms import format_name
from world_machine_experiments.shared.save_metrics import save_metrics
from world_machine_experiments.shared.save_plots import save_plots


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

    generator = torch.Generator()
    generator.manual_seed(0)

    dataset = toy1d_dataloaders["val"].dataset

    if len(dataset) > 32:
        dataset, _ = random_split(
            dataset, [32, len(dataset)-32], generator=generator)

    dataloader = WorldMachineDataLoader(
        dataset, batch_size=32, shuffle=False, generator=generator)

    mg = MetricsGenerator(toy1d_criterion_set)

    _, logits = mg(toy1d_model_trained, dataloader, return_logits=True)

    logits["targets"] = []

    for item in dataloader:
        item["targets"]["state"] = torch.roll(item["inputs"]["state"], -1, 1)
        logits["targets"].append(item["targets"])
        logits["targets"]
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


def toy1d_metrics_sample_plots(toy1d_metrics_sample_logits: dict[str, TensorDict], skip_names: list | None = None) -> dict[str, Figure]:
    if skip_names is None:
        skip_names = []

    time = np.linspace(0, 199, 200, dtype=int)

    batch_size = min(toy1d_metrics_sample_logits["normal"].batch_size[0], 32)

    figures = {}
    for name in toy1d_metrics_sample_logits["normal"].keys():
        if name in skip_names:
            continue

        fig, axs = plt.subplots(4, 8, dpi=300, figsize=(16, 8))
        plt.subplots_adjust(left=None, bottom=None, right=None,
                            top=None, wspace=0.05, hspace=0.05)

        for i in range(batch_size):
            row = i // 8
            column = i % 8

            if name in toy1d_metrics_sample_logits["targets"].keys():
                axs[row, column].plot(toy1d_metrics_sample_logits["targets"]
                                      [name][i], label="Target", color="black")

            axs[row, column].plot(toy1d_metrics_sample_logits["normal"][name][i],
                                  label="Normal", alpha=0.5, color="tab:blue")

            axs[row, column].plot(time, toy1d_metrics_sample_logits["prediction_local"]
                                  [name][i], label="Prediction Local", color="tab:purple")

            axs[row, column].plot(time[:100], toy1d_metrics_sample_logits["use_state"]
                                  [name][i], label="Use State", color="tab:orange")

            axs[row, column].plot(time[100:], toy1d_metrics_sample_logits["prediction"]
                                  [name][i], label="Prediction", color="tab:green")

            axs[row, column].plot(time[100:], toy1d_metrics_sample_logits["prediction_shallow"]
                                  [name][i], label="Prediction Shallow", color="tab:red")

            axs[row, column].set_xticks([])
            axs[row, column].set_yticks([])

            axs[row, column].axvline(100, color="black")

        plt.legend(bbox_to_anchor=(2.5, 4.5), loc='upper right')

        plt.suptitle(f"Metrics Inference Samples - {format_name(name)}")
        # plt.title(name)

        figures["metrics_sample_"+name] = fig

    return figures


save_toy1d_metrics_sample_plots = function_variation({"plots": source(
    "toy1d_metrics_sample_plots")}, "save_toy1d_metrics_sample_plots")(save_plots)
