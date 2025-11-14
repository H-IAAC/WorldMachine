import multiprocessing as mp

import torch
from hamilton import driver
from hamilton_sdk import adapters
from torch.optim import SGD, AdamW
from world_machine_experiments import shared
from world_machine_experiments.toy1d import Channels, parameter_variation

from world_machine.train.scheduler import ChoiceScheduler, UniformScheduler

if __name__ == "__main__":

    mp.set_start_method("spawn")

    d_parameter_variation = driver.Builder().with_modules(
        parameter_variation, shared).build()  # .with_adapter(tracker).build()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_epoch = 2
    output_dir = "test_metrics"

    toy1d_base_args = {"sequence_length": 1000,
                       "n_sequence": 40,
                       "context_size": 200,
                       "state_dimensions": None,
                       "batch_size": 32,
                       "n_epoch": n_epoch,
                       "learning_rate": 5e-4,
                       "weight_decay": 5e-5,
                       "accumulation_steps": 1,
                       "optimizer_class": AdamW,
                       "block_configuration": [Channels.MEASUREMENT, Channels.MEASUREMENT],
                       "device": device,
                       "state_control": "periodic",
                       "positional_encoder_type": "learnable_alibi",
                       "state_activation": "tanh"
                       }

    toy1d_parameter_variation = {
        "Base": {"discover_state": True},
        "M0-100": {"mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "discover_state": True}
    }

    aditional_outputs = ["save_toy1d_metrics",
                         "save_toy1d_metrics_sample_logits",
                         "save_toy1d_metrics_sample_plots",

                         "save_toy1d_mask_sensorial_plot",
                         "save_toy1d_mask_sensorial_metrics",

                         "save_toy1d_autoregressive_metrics"]

    output = d_parameter_variation.execute(["save_toy1d_parameter_variation_plots", "save_toy1d_parameter_variation_mask_sensorial_plots"],

                                           inputs={"base_seed": 42,
                                                   "output_dir": output_dir,
                                                   "n_run": 1,
                                                   "toy1d_base_args": toy1d_base_args,
                                                   "n_worker": 6,
                                                   "toy1d_parameter_variation": toy1d_parameter_variation,
                                                   "aditional_outputs": aditional_outputs
                                                   },
                                           # overrides={
                                           #    "base_dir": output_dir}
                                           )
