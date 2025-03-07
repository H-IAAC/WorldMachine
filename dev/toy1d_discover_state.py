import torch
from hamilton import driver
from hamilton_sdk import adapters
from torch.optim import AdamW
from world_machine_experiments import shared
from world_machine_experiments.toy1d import Dimensions, parameter_variation

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
                       "block_configuration": [Dimensions.NEXT_MEASUREMENT, Dimensions.NEXT_MEASUREMENT],
                       "device": device,
                       "use_state_control": False
                       }

    toy1d_parameter_variation = {
        # "Base": {"discover_state": False},
        # "Discover1": {"discover_state": True, "stable_state_epochs": 1},
        # "Discover1_SS": {"discover_state": True, "stable_state_epochs": 1, "block_configuration": [Dimensions.STATE_AS_SENSORIAL, Dimensions.STATE_AS_SENSORIAL]},
        "Discover1_SS_MASK05": {"discover_state": True, "stable_state_epochs": 1, "block_configuration": [Dimensions.STATE_AS_SENSORIAL, Dimensions.STATE_AS_SENSORIAL], "mask_sensorial_data": 0.5},
        # "Discover2": {"discover_state": True, "stable_state_epochs": 2},
        # "Discover5": {"discover_state": True, "stable_state_epochs": 5},
        # "Discover1_MMMM": {"discover_state": True, "block_configuration": [Dimensions.NEXT_MEASUREMENT, Dimensions.NEXT_MEASUREMENT, Dimensions.NEXT_MEASUREMENT, Dimensions.NEXT_MEASUREMENT]},
        # "Discover1_05LR": {"discover_state": True, "stable_state_epochs": 1, "learning_rate": 0.5*5e-3, "weight_decay": 0.5*5e-4},
    }

    output = d_parameter_variation.execute(["save_toy1d_parameter_variation_plots"],

                                           inputs={"base_seed": 42,
                                                   "output_dir": "toy1d_discover_state3",
                                                   "n_run": 1,
                                                   "toy1d_base_args": toy1d_base_args,
                                                   "n_worker": 9,
                                                   "toy1d_parameter_variation": toy1d_parameter_variation,
                                                   # "custom_plots": {"DiscoverOnly": ["Discover1", "Discover2", "Discover5", "Discover1_MMMM", "Discover1_05LR"],
                                                   #                 "StableState": ["Discover1", "Discover2", "Discover5"]}
                                                   },
                                           # overrides={
                                           #    "base_dir": "toy1d_discover_state"}
                                           )
