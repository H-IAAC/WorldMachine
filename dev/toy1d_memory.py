import torch
from hamilton import driver
from hamilton_sdk import adapters
from torch.optim import AdamW
from world_machine_experiments import shared
from world_machine_experiments.toy1d import Dimensions, parameter_variation

from world_machine.train.scheduler import UniformScheduler

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
    output_dir = "toy1d_memory"

    toy1d_base_args = {"sequence_lenght": 1000,
                       "n_sequence": 10000,
                       "context_size": 200,
                       "state_dimensions": None,
                       "batch_size": 32,
                       "n_epoch": n_epoch,
                       "learning_rate": 5e-3,
                       "weight_decay": 5e-4,
                       "accumulation_steps": 1,
                       "optimizer_class": AdamW,
                       "block_configuration": [Dimensions.NEXT_MEASUREMENT, Dimensions.NEXT_MEASUREMENT],
                       "device": device,
                       "state_control": "periodic",
                       "positional_encoder_type": "learnable_alibi",
                       "state_activation": "tanh"
                       }

    toy1d_parameter_variation = {
        # "Base": {"discover_state": True},
        # "M0-90": {"discover_state": True, "mask_sensorial_data": UniformScheduler(0, 0.9, n_epoch)},
        # "NoDiscover_M0-90": {"discover_state": False, "mask_sensorial_data": UniformScheduler(0, 0.9, n_epoch)}
        # "Break1": {"discover_state": True, "n_segment": 2},
        "Break1_M0-100": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch)},
    }

    aditional_outputs = ["save_toy1d_autoregressive_state_plots",
                         "save_toy1d_autoregressive_positional_encoder_plots",
                         "save_toy1d_autoregressive_state_decoded_plots",
                         "save_toy1d_autoregressive_metrics"]

    output = d_parameter_variation.execute(["save_toy1d_parameter_variation_plots"],

                                           inputs={"base_seed": 42,
                                                   "output_dir": output_dir,
                                                   "n_run": 1,
                                                   "toy1d_base_args": toy1d_base_args,
                                                   "n_worker": 6,
                                                   "toy1d_parameter_variation": toy1d_parameter_variation,
                                                   # "aditional_outputs": aditional_outputs
                                                   },
                                           # overrides={
                                           #    "base_dir": output_dir}
                                           )
