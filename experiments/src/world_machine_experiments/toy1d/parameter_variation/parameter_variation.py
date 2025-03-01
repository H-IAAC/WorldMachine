import glob
import os
from concurrent.futures import (
    Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait)
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from hamilton import driver
from hamilton.function_modifiers import (
    dataloader, datasaver, extract_fields, source, value)
from hamilton_sdk import adapters

from world_machine_experiments import shared
from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.load_train_history import (
    load_train_history)
from world_machine_experiments.shared.parameter_variation_plots import (
    parameter_variation_plots)
from world_machine_experiments.toy1d import multiple


def toy1d_parameter_variation_worker_func(inputs):
    tracker = adapters.HamiltonTracker(
        project_id=1,
        username="EltonCN",
        dag_name="toy1d_multiple_train"
    )

    # .with_adapter(tracker).build()
    d = driver.Builder().with_modules(multiple, shared).build()

    d.execute(["save_multiple_toy1d_train_plots",
               "save_multiple_toy1d_consolidated_train_statistics",
               "save_multiple_toy1d_parameters"],
              inputs=inputs)


@extract_fields({"experiment_paths": dict[str, str], "base_dir": str})
@datasaver()
def save_toy1d_parameter_variation_info(toy1d_base_args: dict[str, Any],
                                        toy1d_parameter_variation: dict[str, dict[str, Any]],
                                        output_dir: str,
                                        n_run: int,
                                        base_seed: int,
                                        n_worker: int = 5) -> dict:

    os.makedirs(output_dir, exist_ok=True)

    executor = ProcessPoolExecutor(n_worker)

    futures: list[Future] = []
    paths = {}
    for run_name in toy1d_parameter_variation:
        toy1d_args = toy1d_base_args.copy()
        toy1d_args.update(toy1d_parameter_variation[run_name])

        run_dir = os.path.join(output_dir, run_name)
        paths[run_name] = run_dir

        inputs = {"base_seed": base_seed,
                  "output_dir": run_dir,
                  "n_run": n_run,
                  "toy1d_args": toy1d_args}

        future = executor.submit(toy1d_parameter_variation_worker_func, inputs)
        futures.append(future)

    with tqdm.tqdm(total=len(futures)) as pbar:
        for future in as_completed(futures):
            future.result()
            pbar.update(1)

    result = {"experiment_paths": paths, "base_dir": output_dir}

    return result


toy1d_load_train_history = function_variation({"history_file_name": value(
    "toy1d_train_history")}, "toy1d_load_train_history")(load_train_history)
toy1d_parameter_variation_plots = function_variation({"train_history": source(
    "toy1d_load_train_history")}, "toy1d_parameter_variation_plots")(parameter_variation_plots)
