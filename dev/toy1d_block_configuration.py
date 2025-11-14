import torch
from hamilton import driver
from hamilton_sdk import adapters
from torch.optim import Adam, AdamW
from world_machine_experiments import shared, toy1d
from world_machine_experiments.toy1d import (
    Channels, base, multiple, parameter_variation)

if __name__ == "__main__":
    tracker = adapters.HamiltonTracker(
        project_id=1,
        username="EltonCN",
        dag_name="toy1d_parameter_variation"
    )

    d_parameter_variation = driver.Builder().with_modules(
        parameter_variation, shared).with_adapter(tracker).build()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    toy1d_base_args = {"sequence_length": 1000,
                       "n_sequence": 10000,
                       "context_size": 200,
                       "state_dimensions": None,
                       "batch_size": 32,
                       "n_epoch": 5,
                       "learning_rate": 5e-3,
                       "weight_decay": 5e-4,
                       "accumulation_steps": 1,
                       "optimizer_class": AdamW,
                       "block_configuration": [Channels.STATE],
                       "device": device,
                       }

    toy1d_parameter_variation = {
        "S": {"block_configuration": [Channels.STATE]},
        "M": {"block_configuration": [Channels.MEASUREMENT]},
        "C": {"block_configuration": [Channels.STATE_CONTROL]},
        "SS": {"block_configuration": [Channels.STATE, Channels.STATE]},
        "MM": {"block_configuration": [Channels.MEASUREMENT, Channels.MEASUREMENT]},
        "SM": {"block_configuration": [Channels.STATE, Channels.MEASUREMENT]},
        "MS": {"block_configuration": [Channels.MEASUREMENT, Channels.STATE]},
        "CC": {"block_configuration": [Channels.STATE_CONTROL, Channels.STATE_CONTROL]},
        "CS": {"block_configuration": [Channels.STATE_CONTROL, Channels.STATE]},
    }

    output = d_parameter_variation.execute(["save_toy1d_parameter_variation_plots"],

                                           inputs={"base_seed": 42,
                                                   "output_dir": "toy1d_block_configuration",
                                                   "n_run": 5,
                                                   "toy1d_base_args": toy1d_base_args,
                                                   "n_worker": 9,
                                                   "toy1d_parameter_variation": toy1d_parameter_variation,
                                                   "custom_plots": {"CvsM": ["C", "CC", "M", "MM"],
                                                                    "Invert": ["SM", "MS"],
                                                                    "WithState": ["CC", "MM", "MS", "CS", "SS"],
                                                                    "BlocksSingle": ["S", "M", "C"],
                                                                    "OnlyState": ["S", "SS"]},
                                                   },
                                           # overrides={
                                           #    "base_dir": "toy1d_block_configuration"}
                                           )
