import os
import time
from typing import Any

import torch
import tqdm
from hamilton import driver
from hamilton.function_modifiers import source, value
from numba import cuda

from world_machine.data import WorldMachineDataset
from world_machine_experiments import shared
from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.load_multiple_metrics import (
    load_multiple_metrics)
from world_machine_experiments.shared.pipeline import save_pipeline
from world_machine_experiments.shared.save_metrics import (
    load_metrics, save_metrics)
from world_machine_experiments.shared.save_parameters import save_parameters
from world_machine_experiments.shared.save_plots import save_plots
from world_machine_experiments.shared.statistics import consolidated_metrics
from world_machine_experiments.shared.train_plots import train_plots
from world_machine_experiments.toy1d import base
from world_machine_experiments.toy1d.base import toy1d_masks_sensory_plots


def multiple_toy1d_trainings_info(n_run: int,
                                  base_seed: int,
                                  output_dir: str,
                                  toy1d_args: dict[str, Any],
                                  aditional_outputs: list[str] | None = None,
                                  minimal: bool = False) -> list[dict]:

    if aditional_outputs is None:
        aditional_outputs = []

    if "device" in toy1d_args:
        device = toy1d_args["device"]

        if "cuda" in device and len(device) > 4:
            cuda.select_device(int(device[5:]))

    d = driver.Builder().with_modules(base, shared).build()

    run_dirs = []
    for i in tqdm.tqdm(range(n_run), unit="run", postfix="multiple_toy1d_training"):
        run_seed = [i, base_seed]
        toy1d_args["seed"] = run_seed

        run_dir = os.path.join(output_dir, f"run_{i}")
        toy1d_args["output_dir"] = run_dir
        run_dirs.append(run_dir)

        if not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=True)

        if minimal:
            final_vars = ["toy1d_train_history",
                          "save_toy1d_model",
                          "save_toy1d_train_history",
                          "toy1d_datasets"]

        else:
            final_vars = ["toy1d_train_history",
                          "save_toy1d_model",
                          "save_toy1d_train_history",
                          "save_toy1d_train_plots",
                          "save_toy1d_prediction_plots",
                          "toy1d_datasets"]

        final_vars += aditional_outputs

        save_pipeline(d, final_vars, "pipeline", run_dir)

        run_check = os.path.join(run_dir, "run_check.txt")
        if not os.path.exists(run_check):
            print(f"Run {i} in {output_dir} starting.")

            outputs = d.execute(final_vars, inputs=toy1d_args)

            for dataset_name in outputs["toy1d_datasets"]:
                dataset: WorldMachineDataset = outputs["toy1d_datasets"][dataset_name]
                dataset.clear_states()

            with open(run_check, "w") as file:
                file.write(str(time.time()))

            del outputs
        else:
            print(f"Run {i} already in {output_dir}. Skipping.")

    results = []
    for run_dir in run_dirs:
        outputs = {}
        outputs["toy1d_train_history"] = load_metrics(
            run_dir, "toy1d_train_history")

        results.append(outputs["toy1d_train_history"])

    return results


save_multiple_toy1d_parameters = function_variation({
    "parameters": source("toy1d_args")},
    "save_multiple_toy1d_parameters")(save_parameters)


# TRAIN HIST

multiple_toy1d_consolidated_train_statistics = function_variation({
    "metrics": source("multiple_toy1d_trainings_info")},
    "multiple_toy1d_consolidated_train_statistics")(consolidated_metrics)

save_multiple_toy1d_consolidated_train_statistics = function_variation({
    "metrics": source("multiple_toy1d_consolidated_train_statistics"),
    "metrics_name": value("toy1d_train_history")},
    "save_multiple_toy1d_consolidated_train_statistics")(save_metrics)

multiple_toy1d_train_plots = function_variation({
    "train_history": source("multiple_toy1d_consolidated_train_statistics")},
    "multiple_toy1d_train_plots")(train_plots)
save_multiple_toy1d_train_plots = function_variation({
    "plots": source("multiple_toy1d_train_plots")},
    "save_multiple_toy1d_train_plots")(save_plots)


# METRICS
multiple_toy1d_metrics = function_variation({
    "metrics_name": value("metrics")},
    "multiple_toy1d_metrics")(load_multiple_metrics)

multiple_toy1d_consolidated_metrics = function_variation({
    "metrics": source(
        "multiple_toy1d_metrics")},
    "multiple_toy1d_consolidated_metrics")(consolidated_metrics)

save_multiple_toy1d_consolidated_metrics = function_variation({
    "metrics": source("multiple_toy1d_consolidated_metrics"),
    "metrics_name": value("toy1d_metrics")},
    "save_multiple_toy1d_consolidated_metrics")(save_metrics)

# MASK SENSORY

multiple_toy1d_mask_sensory_metrics = function_variation({
    "metrics_name": value("mask_sensory_metrics")},
    "multiple_toy1d_mask_sensory_metrics")(load_multiple_metrics)

multiple_toy1d_consolidated_mask_sensory_metrics = function_variation(
    {"metrics": source("multiple_toy1d_mask_sensory_metrics")},
    "multiple_toy1d_consolidated_mask_sensory_metrics")(consolidated_metrics)

save_multiple_toy1d_consolidated_mask_sensory_metrics = function_variation(
    {"metrics": source("multiple_toy1d_consolidated_mask_sensory_metrics"),
     "metrics_name": value("toy1d_mask_sensory_metrics")},
    "save_multiple_toy1d_consolidated_mask_sensory_metrics")(save_metrics)

multiple_toy1d_consolidated_mask_sensory_plots = function_variation({
    "toy1d_mask_sensory_metrics": source("multiple_toy1d_consolidated_mask_sensory_metrics")},
    "multiple_toy1d_consolidated_mask_sensory_plots")(toy1d_masks_sensory_plots)

save_multiple_toy1d_consolidated_mask_sensory_plots = function_variation({
    "plots": source("multiple_toy1d_consolidated_mask_sensory_plots")},
    "save_multiple_toy1d_consolidated_mask_sensory_plots")(save_plots)

# AUTOREGRESSIVE

multiple_toy1d_autoregressive_metrics = function_variation({
    "metrics_name": value("autoregressive_metrics")},
    "multiple_toy1d_autoregressive_metrics")(load_multiple_metrics)

multiple_toy1d_consolidated_autoregressive_metrics = function_variation({
    "metrics": source("multiple_toy1d_autoregressive_metrics")},
    "multiple_toy1d_consolidated_autoregressive_metrics")(consolidated_metrics)

save_multiple_toy1d_consolidated_autoregressive_metrics = function_variation({
    "metrics": source("multiple_toy1d_consolidated_autoregressive_metrics"),
    "metrics_name": value("toy1d_autoregressive_metrics")},
    "save_multiple_toy1d_consolidated_autoregressive_metrics")(save_metrics)
