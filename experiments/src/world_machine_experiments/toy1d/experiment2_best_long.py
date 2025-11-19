import multiprocessing as mp
import os

import torch
from hamilton import driver
from torch.optim import AdamW

from world_machine.train.scheduler import UniformScheduler
from world_machine.train.stages import StateSaveMethod
from world_machine_experiments import shared
from world_machine_experiments.shared.pipeline import save_pipeline
from world_machine_experiments.toy1d import Channels, parameter_variation
from world_machine_experiments.toy1d.specific import experiment2

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
        n_device = 1

    max_jobs_per_device = 12
    n_worker = n_device * max_jobs_per_device

    print("DEVICES", devices)
    print(f"nDevice: {n_device} | nWorker: {n_worker}")

    n_epoch = 1000
    output_dir = "toy1d_experiment2_best_long"

    toy1d_base_args = {"sequence_length": 1000,
                       "n_sequence": 10000,
                       "context_size": 200,
                       "batch_size": 256,
                       "n_epoch": n_epoch,
                       "weight_decay": 5e-5,
                       "accumulation_steps": 1,
                       "state_dimensions": [0],
                       "optimizer_class": AdamW,
                       "device": devices,
                       "state_control": "periodic",
                       "state_activation": "tanh",
                       "discover_state": True,
                       "sensory_train_losses": [Channels.MEASUREMENT],
                       "state_size": 128,
                       "positional_encoder_type": "alibi",
                       "n_attention_head": 4,
                       "n_segment": 2,
                       "fast_forward": True,
                       "stable_state_epochs": 1,
                       "check_input_masks": False,
                       "state_save_method": StateSaveMethod.REPLACE,
                       "mask_sensory_data": UniformScheduler(0, 1, n_epoch),
                       "short_time_recall": {Channels.MEASUREMENT},
                       "recall_stride_past": 3,
                       "recall_stride_future": 3,
                       "recall_n_past": 5,
                       "recall_n_future": 5,
                       "block_configuration": [Channels.MEASUREMENT, Channels.STATE_INPUT],
                       "state_regularizer": None,
                       "train_mse": True,
                       "train_sdtw": False,
                       "noise_config": {"measurement": {"mean": 0.0, "std": 0.1}},
                       "local_chance": None,

                       }

    toy1d_parameter_variation = {
        "StaticLR": {
            "learning_rate": 5e-4
        },
        "CosineAnnealingWithWarmup":
        {
            "learning_rate": 0.001,
            "cosine_annealing": True,
            "cosine_annealing_T_mult": 1,
            "cosine_annealing_T0": 25,
        },
        "CosineAnnealingWithWarmup2x":
        {
            "learning_rate": 0.001,
            "cosine_annealing": True,
            "cosine_annealing_T_mult": 2,
            "cosine_annealing_T0": 25,
        }
    }

    aditional_outputs = ["save_toy1d_metrics",
                         "save_toy1d_metrics_sample_logits",
                         "save_toy1d_metrics_sample_plots",]

    final_vars = ["save_toy1d_parameter_variation_plots"]
    save_pipeline(d_parameter_variation, final_vars,
                  "model_train_pipeline", output_dir)

    d_parameter_variation.execute(final_vars,
                                  inputs={"base_seed": 42,
                                          "output_dir": output_dir,
                                          "n_run": 1,
                                          "toy1d_base_args": toy1d_base_args,
                                          "n_worker": n_worker,
                                          "max_jobs_per_device": max_jobs_per_device,
                                          "toy1d_parameter_variation": toy1d_parameter_variation,
                                          "aditional_outputs": aditional_outputs
                                          }
                                  )

    d_experiment2 = driver.Builder().with_modules(experiment2, shared).build()

    final_vars = ["save_toy1d_samples_plots",
                  "save_prediction_shallow_samples_plots",
                  "save_variation_performance_plots"
                  ]
    save_pipeline(d_experiment2, final_vars,
                  "experiment_pipeline", output_dir)

    d_experiment2.execute(final_vars,

                          inputs={"data_dir": output_dir,
                                  "output_dir": os.path.join(output_dir, "final_results")})
