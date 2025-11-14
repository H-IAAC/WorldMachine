import multiprocessing as mp
import os

import torch
from hamilton import driver
from torch.optim import AdamW
from world_machine_experiments import shared
from world_machine_experiments.toy1d import Channels, parameter_variation
from world_machine_experiments.toy1d.specific import experiment0

from world_machine.train.scheduler import UniformScheduler
from world_machine.train.stages import StateSaveMethod

if __name__ == "__main__":

    mp.set_start_method("spawn")

    d_parameter_variation = driver.Builder().with_modules(
        parameter_variation, shared).build()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_epoch = 100
    output_dir = "cosin_test"

    toy1d_base_args = {"sequence_length": 1000,
                       "n_sequence": 10000,
                       "context_size": 200,
                       "batch_size": 32,
                       "n_epoch": n_epoch,

                       "weight_decay": 5e-5,
                       "accumulation_steps": 1,
                       "state_dimensions": [0],
                       "optimizer_class": AdamW,
                       "block_configuration": [Channels.MEASUREMENT, Channels.MEASUREMENT],
                       "device": device,
                       "state_control": "periodic",
                       "state_activation": "tanh",
                       "discover_state": True,
                       "sensorial_train_losses": [Channels.MEASUREMENT],
                       "state_size": 128,
                       "positional_encoder_type": "alibi",
                       "n_attention_head": 4,
                       "learning_rate": 1e-3,
                       "cosine_annealing": True
                       }

    toy1d_parameter_variation = {
        "CompleteProtocol_T010_mult2": {
            "recall_stride_past": 3, "recall_stride_future": 3, "short_time_recall": {Channels.MEASUREMENT, Channels.STATE_DECODED}, "recall_n_past": 5, "recall_n_future": 5,
            "check_input_masks": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch),
            "n_segment": 2,  "fast_forward": True,
            "noise_config": {"state": {"mean": 0.0, "std": 0.1}, "measurement": {"mean": 0.0, "std": 0.1}},
            "local_chance": 0.25
        },

        "CompleteProtocol_T010_mult1": {
            "recall_stride_past": 3, "recall_stride_future": 3, "short_time_recall": {Channels.MEASUREMENT, Channels.STATE_DECODED}, "recall_n_past": 5, "recall_n_future": 5,
            "check_input_masks": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch),
            "n_segment": 2,  "fast_forward": True,
            "noise_config": {"state": {"mean": 0.0, "std": 0.1}, "measurement": {"mean": 0.0, "std": 0.1}},
            "local_chance": 0.25,
            "cosine_annealing_T_mult": 1,
            "cosine_annealing_T0": 10
        },

        "CompleteProtocol_T020_mult1": {
            "recall_stride_past": 3, "recall_stride_future": 3, "short_time_recall": {Channels.MEASUREMENT, Channels.STATE_DECODED}, "recall_n_past": 5, "recall_n_future": 5,
            "check_input_masks": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch),
            "n_segment": 2,  "fast_forward": True,
            "noise_config": {"state": {"mean": 0.0, "std": 0.1}, "measurement": {"mean": 0.0, "std": 0.1}},
            "local_chance": 0.25,
            "cosine_annealing_T_mult": 1,
            "cosine_annealing_T0": 20
        },


        "CompleteProtocol_T010_mult2_lr5e-4": {
            "recall_stride_past": 3, "recall_stride_future": 3, "short_time_recall": {Channels.MEASUREMENT, Channels.STATE_DECODED}, "recall_n_past": 5, "recall_n_future": 5,
            "check_input_masks": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch),
            "n_segment": 2,  "fast_forward": True,
            "noise_config": {"state": {"mean": 0.0, "std": 0.1}, "measurement": {"mean": 0.0, "std": 0.1}},
            "learning_rate": 5e-4
        },

        "CompleteProtocol_local50": {
            "recall_stride_past": 3, "recall_stride_future": 3, "short_time_recall": {Channels.MEASUREMENT, Channels.STATE_DECODED}, "recall_n_past": 5, "recall_n_future": 5,
            "check_input_masks": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch),
            "n_segment": 2,  "fast_forward": True,
            "noise_config": {"state": {"mean": 0.0, "std": 0.1}, "measurement": {"mean": 0.0, "std": 0.1}},
            "local_chance": 0.5,
            "learning_rate": 5e-4,
            "cosine_annealing": False
        },

        "CompleteProtocol_T020_mult1_lr1e-2": {
            "recall_stride_past": 3, "recall_stride_future": 3, "short_time_recall": {Channels.MEASUREMENT, Channels.STATE_DECODED}, "recall_n_past": 5, "recall_n_future": 5,
            "check_input_masks": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch),
            "n_segment": 2,  "fast_forward": True,
            "noise_config": {"state": {"mean": 0.0, "std": 0.1}, "measurement": {"mean": 0.0, "std": 0.1}},
            "local_chance": 0.25,
            "cosine_annealing_T_mult": 1,
            "cosine_annealing_T0": 20,
            "learning_rate": 1e-2,
        },

        "CompleteProtocol_T020_mult2": {
            "recall_stride_past": 3, "recall_stride_future": 3, "short_time_recall": {Channels.MEASUREMENT, Channels.STATE_DECODED}, "recall_n_past": 5, "recall_n_future": 5,
            "check_input_masks": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch),
            "n_segment": 2,  "fast_forward": True,
            "noise_config": {"state": {"mean": 0.0, "std": 0.1}, "measurement": {"mean": 0.0, "std": 0.1}},
            "local_chance": 0.25,
            "cosine_annealing_T_mult": 2,
            "cosine_annealing_T0": 20
        },

        "Best_T020_mult1": {
            "block_configuration": [Channels.MEASUREMENT, Channels.STATE_INPUT],
            "n_segment": 1,
            "fast_forward": False,
            "stable_state_epochs": 1,
            "check_input_masks": True,
            "state_save_method": StateSaveMethod.REPLACE,
            "mask_sensorial_data": UniformScheduler(0, 1, n_epoch),
            "short_time_recall": {
                Channels.MEASUREMENT
            },
            "recall_stride_past": 3,
            "recall_stride_future": 3,
            "recall_n_past": 5,
            "recall_n_future": 5,
            "train_mse": True,
            "train_sdtw": False,
            "noise_config": {
                "state": {
                    "mean": 0.0,
                    "std": 0.1
                },
                "measurement": {
                    "mean": 0.0,
                    "std": 0.1
                }
            },
            "local_chance": 0.25,
            "cosine_annealing_T_mult": 1,
            "cosine_annealing_T0": 20
        },

        "CompleteProtocol_lr1e-3": {
            "recall_stride_past": 3, "recall_stride_future": 3, "short_time_recall": {Channels.MEASUREMENT, Channels.STATE_DECODED}, "recall_n_past": 5, "recall_n_future": 5,
            "check_input_masks": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch),
            "n_segment": 2,  "fast_forward": True,
            "noise_config": {"state": {"mean": 0.0, "std": 0.1}, "measurement": {"mean": 0.0, "std": 0.1}},
            "local_chance": 0.25,
            "learning_rate": 1e-3,
            "cosine_annealing": False
        },

        "CompleteProtocol_T025_mult1_8Head": {
            "recall_stride_past": 3, "recall_stride_future": 3, "short_time_recall": {Channels.MEASUREMENT, Channels.STATE_DECODED}, "recall_n_past": 5, "recall_n_future": 5,
            "check_input_masks": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch),
            "n_segment": 2,  "fast_forward": True,
            "noise_config": {"state": {"mean": 0.0, "std": 0.1}, "measurement": {"mean": 0.0, "std": 0.1}},
            "local_chance": 0.25,
            "cosine_annealing_T_mult": 1,
            "cosine_annealing_T0": 25,
            "n_attention_head": 8
        },

        "CompleteProtocol_T020_mult1_seg5": {
            "recall_stride_past": 3, "recall_stride_future": 3, "short_time_recall": {Channels.MEASUREMENT, Channels.STATE_DECODED}, "recall_n_past": 5, "recall_n_future": 5,
            "check_input_masks": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch),
            "n_segment": 5,  "fast_forward": True,
            "noise_config": {"state": {"mean": 0.0, "std": 0.1}, "measurement": {"mean": 0.0, "std": 0.1}},
            "local_chance": 0.25,
            "cosine_annealing_T_mult": 1,
            "cosine_annealing_T0": 20
        },
    }

    aditional_outputs = ["save_toy1d_metrics",
                         "save_toy1d_metrics_sample_logits",
                         "save_toy1d_metrics_sample_plots"]

    d_parameter_variation.execute(["save_toy1d_parameter_variation_plots"],
                                  inputs={"base_seed": 42,
                                          "output_dir": output_dir,
                                          "n_run": 1,
                                          "toy1d_base_args": toy1d_base_args,
                                          "n_worker": 10,
                                          "toy1d_parameter_variation": toy1d_parameter_variation,
                                          "aditional_outputs": aditional_outputs
                                          }
                                  )
