import os
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
import tqdm
from hamilton import driver
from hamilton.function_modifiers import (
    datasaver, extract_fields, source, value)

from world_machine_experiments import shared
from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.parameter_variation_plots import (
    parameter_variation_plots)
from world_machine_experiments.shared.save_metrics import load_multiple_metrics
from world_machine_experiments.shared.save_plots import save_plots
from world_machine_experiments.toy1d import multiple


def worker_initializer(lock, num_threads: int | None = None):
    tqdm.tqdm.set_lock(lock)

    if num_threads is not None:
        torch.set_num_threads(num_threads)


def toy1d_parameter_variation_worker_func(inputs):
    d = driver.Builder().with_modules(multiple, shared).build()

    outputs = ["save_multiple_toy1d_train_plots",
               "save_multiple_toy1d_consolidated_train_statistics",
               "save_multiple_toy1d_parameters",
               ]

    if inputs["aditional_outputs"] is not None:
        if "save_toy1d_mask_sensorial_metrics" in inputs["aditional_outputs"]:
            outputs += ["save_multiple_toy1d_consolidated_mask_sensorial_metrics",
                        "save_multiple_toy1d_consolidated_mask_sensorial_plots"]

        if "save_toy1d_metrics" in inputs["aditional_outputs"]:
            outputs.append("save_multiple_toy1d_consolidated_metrics")

        if "save_toy1d_autoregressive_metrics" in inputs["aditional_outputs"]:
            outputs.append(
                "save_multiple_toy1d_consolidated_autoregressive_metrics")

    d.execute(outputs,
              inputs=inputs)

    if "device" in inputs["toy1d_args"]:
        return inputs["toy1d_args"]["device"]
    return None


@extract_fields({"experiment_paths": dict[str, str], "base_dir": str})
@datasaver()
def save_toy1d_parameter_variation_info(toy1d_base_args: dict[str, Any],
                                        toy1d_parameter_variation: dict[str, dict[str, Any]],
                                        output_dir: str,
                                        n_run: int,
                                        base_seed: int,
                                        n_worker: int = 5,
                                        aditional_outputs: list[str] | None = None,
                                        max_jobs_per_device: int | None = None) -> dict:

    os.makedirs(output_dir, exist_ok=True)

    lock = mp.RLock()

    n_thread_per_worker = mp.cpu_count()//n_worker

    executor = ProcessPoolExecutor(n_worker,  
                                   initializer=worker_initializer, 
                                   initargs=(lock, n_thread_per_worker), 
                                   max_tasks_per_child=10)

    devices = []
    if "device" in toy1d_base_args:
        if isinstance(toy1d_base_args["device"], list):
            devices += toy1d_base_args["device"]
        else:
            devices.append(toy1d_base_args["device"])
    else:
        devices.append(None)

    jobs_per_device = np.zeros(len(devices), int)

    futures: list[Future] = []
    paths = {}

    names = list(toy1d_parameter_variation.keys())

    pbar = tqdm.tqdm(total=len(toy1d_parameter_variation),
                     desc="Parameter Variation")
    n_finish = 0

    while len(names) != 0:
        run_name = names.pop()

        device_index = np.argmin(jobs_per_device)

        if (max_jobs_per_device is not None and
                jobs_per_device[device_index] >= max_jobs_per_device):

            future = next(as_completed(futures))
            futures.remove(future)
            pbar.update(1)
            n_finish += 1

            print(f"{n_finish} finished")

            device = future.result()
            done_device_index = devices.index(device)
            jobs_per_device[done_device_index] -= 1

            device_index = np.argmin(jobs_per_device)

        toy1d_args = toy1d_base_args.copy()
        toy1d_args.update(toy1d_parameter_variation[run_name])

        device = devices[device_index]
        if device is not None:
            toy1d_args["device"] = device

        run_dir = os.path.join(output_dir, run_name)
        paths[run_name] = run_dir

        inputs = {"base_seed": base_seed,
                  "output_dir": run_dir,
                  "n_run": n_run,
                  "toy1d_args": toy1d_args,
                  "aditional_outputs": aditional_outputs}

        future = executor.submit(toy1d_parameter_variation_worker_func, inputs)
        futures.append(future)

        jobs_per_device[device_index] += 1

    for future in as_completed(futures):
        future.result()
        pbar.update(1)
        n_finish += 1
        print(f"{n_finish} finished")

    pbar.close()

    result = {"experiment_paths": paths, "base_dir": output_dir}

    return result


toy1d_load_train_history = function_variation({
    "metrics_name": value("toy1d_train_history"),
    "output_dir": source("base_dir")},
    "toy1d_load_train_history")(load_multiple_metrics)

toy1d_parameter_variation_plots = function_variation({
    "train_history": source("toy1d_load_train_history")},
    "toy1d_parameter_variation_plots")(parameter_variation_plots)


# PLOT
save_toy1d_parameter_variation_plots = function_variation({
    "plots": source("toy1d_parameter_variation_plots")},
    "save_toy1d_parameter_variation_plots")(save_plots)

# MASK SENSORIAL
toy1d_load_mask_sensorial_metrics = function_variation({
    "metrics_name": value("toy1d_mask_sensorial_metrics"),
    "output_dir": source("base_dir")},
    "toy1d_load_mask_sensorial_metrics")(load_multiple_metrics)

toy1d_parameter_variation_mask_sensorial_plots = function_variation({
    "train_history": source("toy1d_load_mask_sensorial_metrics"),
    "x_axis": value("mask_sensorial_percentage"),
    "plot_prefix": value("mask_sensorial"),
    "series_names": value([])},
    "toy1d_parameter_variation_mask_sensorial_plots")(parameter_variation_plots)

save_toy1d_parameter_variation_mask_sensorial_plots = function_variation({
    "plots": source("toy1d_parameter_variation_mask_sensorial_plots")},
    "save_toy1d_parameter_variation_mask_sensorial_plots")(save_plots)
