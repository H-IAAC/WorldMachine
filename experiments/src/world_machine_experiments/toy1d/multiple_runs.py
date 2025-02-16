import os
from typing import Any

import numpy as np
from hamilton import driver
from hamilton_sdk import adapters


from world_machine_experiments import toy1d, shared


def multiple_toy1d_trainings_info(n_run:int, 
                             base_seed:int, 
                             output_dir:str,
                             toy1d_args:dict[str, Any]) -> list[dict[str, np.ndarray]]:
    tracker = adapters.HamiltonTracker(
        project_id=1,
        username="EltonCN",
        dag_name="toy1d_train"
    )

    d = driver.Builder().with_modules(toy1d, shared).with_adapter(tracker).build()

    results = []
    for i in range(n_run):
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


def multiple_toy1d_consolidated_train_statistics(multiple_toy1d_trainings_info:list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    
    consolidated = {}
    for name in multiple_toy1d_trainings_info[0]:
        entry_from_all = []
        for i in range(len(multiple_toy1d_trainings_info)):
            entry_from_all.append(multiple_toy1d_trainings_info[i][name])
        
        consolidated[name] = np.mean(entry_from_all, axis=0)
        consolidated[name+"_std"] = np.std(entry_from_all, axis=0)


    return consolidated