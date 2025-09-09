import multiprocessing as mp

import torch
from hamilton import driver
from hamilton_sdk import adapters
from torch.optim import SGD, AdamW

from world_machine.train.scheduler import UniformScheduler
from world_machine_experiments import shared
from world_machine_experiments.toy1d import Dimensions, parameter_variation

if __name__ == "__main__":

    mp.set_start_method("spawn")

    d_parameter_variation = driver.Builder().with_modules(
        parameter_variation, shared).build()

    devices = []
    if torch.cuda.is_available():
        n_device = torch.cuda.device_count()

        for i in range(n_device):
            devices.append(f"cuda:{i}")

    else:
        devices.append('cpu')

    n_epoch = 1  # 100
    output_dir = "toy1d_experiment1_"

    toy1d_base_args = {"sequence_lenght": 1000,
                       "n_sequence": 50,  # 10000,
                       "context_size": 200,
                       "batch_size": 32,
                       "n_epoch": n_epoch,
                       "learning_rate": 5e-4,
                       "weight_decay": 5e-5,
                       "accumulation_steps": 1,
                       "state_dimensions": [0],
                       "optimizer_class": AdamW,
                       "block_configuration": [Dimensions.NEXT_MEASUREMENT, Dimensions.NEXT_MEASUREMENT],
                       "device": devices,
                       "state_control": "periodic",
                       "state_activation": "tanh",
                       "discover_state": True,
                       "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT],
                       "state_size": 128,
                       "positional_encoder_type": "alibi",
                       "n_attention_head": 4
                       }

    toy1d_parameter_variation = {
        "Base": {},
        "Base1": {},
        "Base2": {},
        "Base3": {},
        "Base4": {},
        "Base5": {},
        "Base6": {},
        "Base7": {},
        "Base8": {},
    }

    aditional_outputs = []  # ["save_toy1d_metrics",
    # "save_toy1d_metrics_sample_logits",
    # "save_toy1d_metrics_sample_plots",
#
    # "save_toy1d_autoregressive_metrics"]

    output = d_parameter_variation.execute(["save_toy1d_parameter_variation_plots"],
                                           inputs={"base_seed": 42,
                                                   "output_dir": output_dir,
                                                   "n_run": 1,
                                                   "toy1d_base_args": toy1d_base_args,
                                                   "n_worker": 6,
                                                   "max_jobs_per_device": 2,
                                                   "toy1d_parameter_variation": toy1d_parameter_variation,
                                                   "aditional_outputs": aditional_outputs
                                                   }
                                           )
