import torch
from hamilton import driver
from hamilton_sdk import adapters
from torch.optim import Adam, AdamW
from world_machine_experiments import shared, toy1d
from world_machine_experiments.toy1d import (
    Channels, base, multiple, parameter_variation)

from world_machine.train import UniformScheduler

if __name__ == "__main__":
    tracker = adapters.HamiltonTracker(
        project_id=1,
        username="EltonCN",
        dag_name="toy1d_parameter_variation"
    )

    d_parameter_variation = driver.Builder().with_modules(
        parameter_variation, shared).with_adapter(tracker).build()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_epoch = 5

    toy1d_base_args = {"sequence_length": 1000,
                       "n_sequence": 10000,
                       "context_size": 200,
                       "state_dimensions": None,
                       "batch_size": 32,
                       "n_epoch": n_epoch,
                       "learning_rate": 5e-3,
                       "weight_decay": 5e-4,
                       "accumulation_steps": 1,
                       "optimizer_class": AdamW,
                       "block_configuration": [Channels.MEASUREMENT],
                       "device": device,
                       }

    toy1d_parameter_variation = {
        "000": {"mask_sensorial_data": 0.0},
        "025": {"mask_sensorial_data": 0.25},
        "05": {"mask_sensorial_data": 0.5},
        "075": {"mask_sensorial_data": 0.75},
        "100": {"mask_sensorial_data": 1.0},
        "000-075": {"mask_sensorial_data": UniformScheduler(0, 0.75, n_epoch)},
        "000-100": {"mask_sensorial_data": UniformScheduler(0, 1.0, n_epoch)},
        "000-075L": {"mask_sensorial_data": UniformScheduler(0, 0.75, n_epoch), "learn_sensorial_mask": True},
        "05L": {"mask_sensorial_data": 0.5, "learn_sensorial_mask": True},
    }

    aditional_outputs = ["save_toy1d_masks_sensorial_plot",
                         "save_toy1d_mask_sensorial_metrics"]

    output = d_parameter_variation.execute(["save_toy1d_parameter_variation_plots",
                                            "save_toy1d_parameter_variation_mask_sensorial_plots"],

                                           inputs={"base_seed": 42,
                                                   "output_dir": "toy1d_mask_sensorial",
                                                   "n_run": 5,
                                                   "toy1d_base_args": toy1d_base_args,
                                                   "n_worker": 5,
                                                   "toy1d_parameter_variation": toy1d_parameter_variation,
                                                   "aditional_outputs": aditional_outputs},
                                           # overrides={
                                           #    "base_dir": "toy1d_mask_sensorial"}
                                           )
