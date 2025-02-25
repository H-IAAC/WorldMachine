import os
from typing import Any

import numpy as np
import tqdm
from hamilton import driver
from hamilton.function_modifiers import source
from hamilton_sdk import adapters

from world_machine_experiments import shared, toy1d
from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.save_parameters import save_parameters
from world_machine_experiments.toy1d import base


def multiple_toy1d_trainings_info(n_run: int,
                                  base_seed: int,
                                  output_dir: str,
                                  toy1d_args: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    tracker = adapters.HamiltonTracker(
        project_id=1,
        username="EltonCN",
        dag_name="toy1d_train"
    )

    # .with_adapter(tracker).build()
    d = driver.Builder().with_modules(base, shared).build()

    results = []
    for i in tqdm.tqdm(range(n_run), unit="run", postfix="multiple_toy1d_training"):
        run_seed = [i, base_seed]
        toy1d_args["seed"] = run_seed

        run_dir = os.path.join(output_dir, f"run_{i}")
        toy1d_args["output_dir"] = run_dir

        if not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=True)

        outputs = d.execute(["toy1d_train_history",
                             "save_toy1d_model",
                             "save_toy1d_train_history",
                             "save_toy1d_train_plots",
                             "save_toy1d_prediction_plots"], inputs=toy1d_args)

        results.append(outputs["toy1d_train_history"])

    return results


save_multiple_toy1d_parameters = function_variation(
    {"parameters": source("toy1d_args")}, "save_multiple_toy1d_parameters")(save_parameters)
