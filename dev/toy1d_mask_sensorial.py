import torch
from hamilton import driver
from hamilton_sdk import adapters
from torch.optim import Adam, AdamW
from world_machine_experiments import shared, toy1d
from world_machine_experiments.toy1d import (
    Dimensions, base, multiple, parameter_variation)

if __name__ == "__main__":
    tracker = adapters.HamiltonTracker(
        project_id=1,
        username="EltonCN",
        dag_name="toy1d_parameter_variation"
    )

    d_parameter_variation = driver.Builder().with_modules(
        parameter_variation, shared).with_adapter(tracker).build()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    toy1d_base_args = {"sequence_lenght": 1000,
                       "n_sequence": 10000,
                       "context_size": 200,
                       "state_dimensions": None,
                       "batch_size": 32,
                       "n_epoch": 5,
                       "learning_rate": 5e-3,
                       "weight_decay": 5e-4,
                       "accumulation_steps": 1,
                       "optimizer_class": AdamW,
                       "block_configuration": [Dimensions.NEXT_MEASUREMENT],
                       "device": device,
                       }

    toy1d_parameter_variation = {
        "none_NEW": {"mask_sensorial_data": None},
        # "0": {"mask_sensorial_data": 0.0},
        # "025": {"mask_sensorial_data": 0.25},
        # "05": {"mask_sensorial_data": 0.5},
        # "075": {"mask_sensorial_data": 0.75},
        "1_NEW": {"mask_sensorial_data": 1.0},
        # "05_NEW": {"mask_sensorial_data": 0.5}
    }

    output = d_parameter_variation.execute(["save_toy1d_parameter_variation_plots"],

                                           inputs={"base_seed": 42,
                                                   "output_dir": "toy1d_mask_sensorial",
                                                   "n_run": 5,
                                                   "toy1d_base_args": toy1d_base_args,
                                                   "n_worker": 6,
                                                   "toy1d_parameter_variation": toy1d_parameter_variation},
                                           # overrides={
                                           #    "base_dir": "toy1d_mask_sensorial"}
                                           )
