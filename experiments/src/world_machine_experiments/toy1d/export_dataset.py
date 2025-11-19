import multiprocessing as mp
import os

import torch
from hamilton import driver
from torch.optim import AdamW

from world_machine.train.scheduler import UniformScheduler
from world_machine_experiments import shared
from world_machine_experiments.shared.pipeline import save_pipeline
from world_machine_experiments.toy1d import Channels, base, parameter_variation
from world_machine_experiments.toy1d.specific import experiment0

if __name__ == "__main__":
    output_dir = "toy1d_dataset"
    os.makedirs(output_dir, exist_ok=True)

    d = driver.Builder().with_modules(base, shared).build()
    final_vars = ["save_toy1d_datasets"]
    save_pipeline(d, final_vars, "pipeline", output_dir)

    args = {"sequence_length": 1000,
            "n_sequence": 10000,
            "context_size": 200,
            "state_control": "periodic",
            "state_dimensions": [0],
            }

    base_seed = 42
    for seed in range(15):
        run_dir = os.path.join(output_dir, f"seed_{seed}")

        args.update({"seed": [seed, base_seed],
                     "output_dir": run_dir})

        d.execute(final_vars, inputs=args)
