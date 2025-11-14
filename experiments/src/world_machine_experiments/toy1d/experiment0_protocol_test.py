import multiprocessing as mp
import os

import torch
from hamilton import driver
from torch.optim import AdamW

from world_machine.train.scheduler import UniformScheduler
from world_machine_experiments import shared
from world_machine_experiments.shared.pipeline import save_pipeline
from world_machine_experiments.toy1d import Channels, parameter_variation
from world_machine_experiments.toy1d.specific import experiment0

if __name__ == "__main__":

    mp.set_start_method("spawn")

    d_parameter_variation = driver.Builder().with_modules(
        parameter_variation, shared).build()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_epoch = 100
    output_dir = "toy1d_experiment0_protocol_test"

    toy1d_base_args = {"sequence_length": 1000,
                       "n_sequence": 10000,
                       "context_size": 200,
                       "batch_size": 32,
                       "n_epoch": n_epoch,
                       "learning_rate": 1e-3,
                       "cosine_annealing": True,
                       "cosine_annealing_T_mult": 1,
                       "cosine_annealing_T0": 25,
                       "weight_decay": 5e-5,
                       "accumulation_steps": 1,
                       "state_dimensions": [0],
                       "optimizer_class": AdamW,
                       "block_configuration": [Channels.MEASUREMENT, Channels.MEASUREMENT],
                       "device": device,
                       "state_control": "periodic",
                       "state_activation": "tanh",
                       "discover_state": True,
                       "sensory_train_losses": [Channels.MEASUREMENT],
                       "state_size": 128,
                       "positional_encoder_type": "alibi",
                       "n_attention_head": 4
                       }

    toy1d_parameter_variation = {
        "Base": {},
        "SensoryMask": {"mask_sensory_data": UniformScheduler(0, 1, n_epoch)},

        "CompleteProtocol": {
            "recall_stride_past": 3, "recall_stride_future": 3, "short_time_recall": {Channels.MEASUREMENT, Channels.STATE_DECODED}, "recall_n_past": 5, "recall_n_future": 5,
            "check_input_masks": True, "mask_sensory_data": UniformScheduler(0, 1, n_epoch),
            "n_segment": 2,  "fast_forward": True,
            "noise_config": {"state": {"mean": 0.0, "std": 0.1}, "measurement": {"mean": 0.0, "std": 0.1}},
            "local_chance": 0.25
        },
    }

    aditional_outputs = ["save_toy1d_metrics",
                         "save_toy1d_metrics_sample_logits",
                         "save_toy1d_metrics_sample_plots",

                         "save_toy1d_mask_sensory_plot",
                         "save_toy1d_mask_sensory_metrics",

                         "save_toy1d_autoregressive_metrics"]

    outputs = ["save_toy1d_parameter_variation_plots",
               "save_toy1d_parameter_variation_mask_sensory_plots"]

    save_pipeline(d_parameter_variation, outputs,
                  "model_train_pipeline", output_dir)
    d_parameter_variation.execute(outputs,
                                  inputs={"base_seed": 42,
                                          "output_dir": output_dir,
                                          "n_run": 15,
                                          "toy1d_base_args": toy1d_base_args,
                                          "n_worker": 3,
                                          "toy1d_parameter_variation": toy1d_parameter_variation,
                                          "aditional_outputs": aditional_outputs
                                          }
                                  )

    d_experiment0 = driver.Builder().with_modules(experiment0, shared).build()

    outputs = ["save_train_plots",
               "save_metrics_box_plots",
               "save_metrics_bar_plots",
               "save_samples_plots",
               "save_state_analysis_plots",
               "save_target_state09_correlation_plot"]
    save_pipeline(d_experiment0, outputs, "experiment_pipeline", output_dir)
    d_experiment0.execute(outputs,
                          inputs={"data_dir": output_dir,
                                  "output_dir": os.path.join(output_dir, "final_results")})
