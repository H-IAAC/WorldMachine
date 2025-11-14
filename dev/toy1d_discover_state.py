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
    output_dir = "toy1d_discover_state"
    if long:
        n_epoch = 100
        output_dir = "toy1d_discover_state_long"

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
                       "positional_encoder_type": "learnable_alibi",
                       "state_activation": "tanh"
                       }

    toy1d_parameter_variation = {
        "Base_MM": {"discover_state": False},
        "Discover1_MM": {"discover_state": True, "stable_state_epochs": 1},
        "Discover1_MM_TRAIN_M": {"discover_state": True, "stable_state_epochs": 1, "block_configuration": [Channels.MEASUREMENT, Channels.MEASUREMENT], "sensorial_train_losses": {Channels.MEASUREMENT}},
        "Discover1_MM_TRAIN_M_3M": {"discover_state": True, "stable_state_epochs": 1, "block_configuration": [Channels.MEASUREMENT, Channels.MEASUREMENT], "sensorial_train_losses": {Channels.MEASUREMENT}, "measurement_size": 3},
        "Discover1_DD": {"discover_state": True, "stable_state_epochs": 1, "block_configuration": [Channels.STATE_DECODED, Channels.STATE_DECODED]},
        "Discover1_DM": {"discover_state": True, "stable_state_epochs": 1, "block_configuration": [Channels.STATE_DECODED, Channels.MEASUREMENT]},
        "Discover1_DM_TRAIN_M": {"discover_state": True, "stable_state_epochs": 1, "block_configuration": [Channels.STATE_DECODED, Channels.MEASUREMENT], "sensorial_train_losses": {Channels.MEASUREMENT}},
        "Discover1_DD_MASK05": {"discover_state": True, "stable_state_epochs": 1, "block_configuration": [Channels.STATE_DECODED, Channels.STATE_DECODED], "mask_sensorial_data": 0.5},
        "Discover1_DD_MASKLINEARD": {"discover_state": True, "stable_state_epochs": 1, "block_configuration": [Channels.STATE_DECODED, Channels.STATE_DECODED], "mask_sensorial_data": LinearScheduler(0.0, 1.0, n_epoch)},
        "Discover1_DM_MASKLINEARD": {"discover_state": True, "stable_state_epochs": 1, "block_configuration": [Channels.STATE_DECODED, Channels.MEASUREMENT], "mask_sensorial_data": {"state_decoded": LinearScheduler(0.0, 1.0, n_epoch)}},
        "Discover2_MM": {"discover_state": True, "stable_state_epochs": 2},
        "Discover2_DD": {"discover_state": True, "stable_state_epochs": 2,  "block_configuration": [Channels.STATE_DECODED, Channels.STATE_DECODED]},
    }

    output = d_parameter_variation.execute(["save_toy1d_parameter_variation_plots"],

                                           inputs={"base_seed": 42,
                                                   "output_dir": output_dir,
                                                   "n_run": 1,
                                                   "toy1d_base_args": toy1d_base_args,
                                                   "n_worker": 6,
                                                   "toy1d_parameter_variation": toy1d_parameter_variation,
                                                   "custom_plots": {"MaskLinear": ["Discover1_DD_MASKLINEARD", "Discover1_DM_MASKLINEARD", "Base_MM", "Discover1_DD", "Discover1_DM"],
                                                                    "StableState": ["Discover1_DD", "Discover1_MM", "Discover2_MM", "Discover2_DD"],
                                                                    "TrainM": ["Base_MM", "Discover1_MM", "Discover1_MM_TRAIN_M", "Discover1_DM", "Discover1_DM_TRAIN_M", "Discover1_MM_TRAIN_M_3M"],
                                                                    "DxM": ["Base_MM", "Discover1_MM", "Discover1_DD"]}
                                                   },
                                           # overrides={
                                           #    "base_dir": output_dir}
                                           )
