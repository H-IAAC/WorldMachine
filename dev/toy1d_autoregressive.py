import os

import torch
from hamilton import driver
from hamilton_sdk import adapters
from torch.optim import AdamW
from world_machine_experiments import shared
from world_machine_experiments.toy1d import Channels, parameter_variation

from world_machine.train.scheduler import LinearScheduler

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
    output_dir = "toy1d_autoregressive"

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
                       "positional_encoder_type": "sine",
                       "state_activation": None
                       }

    toy1d_parameter_variation = {
        "Base": {},
        "Clamp": {"state_activation": "clamp"},
        "RemovePE": {"remove_positional_encoding": True},
        "NoPE": {"positional_encoder_type": None},
        "NoPE_Clamp": {"positional_encoder_type": None, "state_activation": "clamp"},
        "NoPE_Tanh": {"positional_encoder_type": None, "state_activation": "tanh"},
        "RemovePE_Tanh": {"remove_positional_encoding": True, "state_activation": "tanh"},
        "Tanh": {"state_activation": "tanh"},
        "Albi_Tanh": {"positional_encoder_type": "alibi", "state_activation": "tanh"},
        "LAlbi_Tanh": {"positional_encoder_type": "learnable_alibi", "state_activation": "tanh"}
    }

    experiments_paths = {}
    for experiment_name in toy1d_parameter_variation:
        experiments_paths[experiment_name] = os.path.join(
            output_dir, experiment_name)

    aditional_outputs = ["save_toy1d_autoregressive_state_plots",
                         "save_toy1d_autoregressive_positional_encoder_plots",
                         "save_toy1d_autoregressive_state_decoded_plots",
                         "save_toy1d_autoregressive_metrics"]

    output = d_parameter_variation.execute(["save_toy1d_parameter_variation_plots", "save_variation_autoregressive_metrics"],

                                           inputs={"base_seed": 42,
                                                   "output_dir": output_dir,
                                                   "n_run": 1,
                                                   "toy1d_base_args": toy1d_base_args,
                                                   "n_worker": 7,
                                                   "toy1d_parameter_variation": toy1d_parameter_variation,
                                                   "aditional_outputs": aditional_outputs
                                                   },
                                           overrides={
        "base_dir": output_dir,
        "experiment_paths": experiments_paths
    }
    )
