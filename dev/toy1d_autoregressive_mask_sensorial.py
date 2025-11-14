import os

import torch
from hamilton import driver
from hamilton_sdk import adapters
from torch.optim import AdamW
from world_machine_experiments import shared
from world_machine_experiments.toy1d import Channels, parameter_variation

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

    long = False

    n_epoch = 5
    output_dir = "toy1d_autoregressive_mask_sensorial"

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
                       "block_configuration": [Channels.MEASUREMENT, Channels.MEASUREMENT],
                       "device": device,
                       "state_control": False,
                       "discover_state": True,
                       "stable_state_epochs": 1,
                       "remove_positional_encoding": False,
                       "positional_encoder_type": "learnable_alibi",
                       "state_activation": "tanh"
                       }

    toy1d_parameter_variation = {
        "000-100": {"mask_sensorial_data": UniformScheduler(0, 1.0, n_epoch)},
        # "050": {"mask_sensorial_data": 0.5},
        # "000": {"mask_sensorial_data": 0.0},
        # "000-100L": {"mask_sensorial_data": UniformScheduler(0, 1.0, n_epoch), "learn_sensorial_mask": True},
        # "05L": {"mask_sensorial_data": 0.5, "learn_sensorial_mask": True},
    }

    experiments_paths = {}
    for experiment_name in toy1d_parameter_variation:
        experiments_paths[experiment_name] = os.path.join(
            output_dir, experiment_name)

    aditional_outputs = ["save_toy1d_autoregressive_state_plots",
                         "save_toy1d_autoregressive_positional_encoder_plots",
                         "save_toy1d_autoregressive_state_decoded_plots",
                         "save_toy1d_autoregressive_metrics",
                         "save_toy1d_masks_sensorial_plot",
                         "save_toy1d_mask_sensorial_metrics"
                         ]

    output = d_parameter_variation.execute(["save_toy1d_parameter_variation_plots", "save_variation_autoregressive_metrics"],

                                           inputs={"base_seed": 42,
                                                   "output_dir": output_dir,
                                                   "n_run": 1,
                                                   "toy1d_base_args": toy1d_base_args,
                                                   "n_worker": 7,
                                                   "toy1d_parameter_variation": toy1d_parameter_variation,
                                                   "aditional_outputs": aditional_outputs
                                                   },
                                           # overrides={"base_dir": output_dir,
                                           #           "experiment_paths": experiments_paths
                                           #           }
                                           )
