import multiprocessing as mp
import os
import pickle

import torch
from hamilton import driver
from torch.optim import AdamW

from world_machine.train.scheduler import UniformScheduler
from world_machine.train.stages import StateSaveMethod
from world_machine_experiments import shared
from world_machine_experiments.shared.save_parameters import make_model
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

    max_jobs_per_device = 10
    n_worker = n_device * max_jobs_per_device

    output_dir = "toy1d_experiment1_configuration_test"
    n_epoch = 100

    toy1d_base_args = {"sequence_length": 1000,
                       "n_sequence": 10000,
                       "context_size": 200,
                       "batch_size": 256,
                       "n_epoch": n_epoch,
                       "learning_rate": 5e-4,
                       "weight_decay": 5e-5,
                       "accumulation_steps": 1,
                       "state_dimensions": [0],
                       "optimizer_class": AdamW,
                       "device": devices,
                       "state_control": "periodic",
                       "discover_state": True,
                       "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT],
                       "state_size": 128,
                       "positional_encoder_type": "alibi",
                       "n_attention_head": 4,
                       }

    n_variation = 0
    configurations = {}
    for n_segment in [1, 2]:
        fast_forward_choices = [False]
        if n_segment > 1:
            fast_forward_choices = [True, False]

        for fast_forward in fast_forward_choices:
            for stable_state_epochs in [1]:
                for check_input_masks in [True, False]:
                    for state_save_method in [StateSaveMethod.MEAN, StateSaveMethod.REPLACE]:
                        for mask_sensorial_data in [UniformScheduler(0, 1, n_epoch)]:
                            for short_time_recall in [set(), {Dimensions.NEXT_MEASUREMENT}]:

                                recall_n_past_choices = [0]
                                recall_stride_choices = [0]
                                if len(short_time_recall) > 0:
                                    recall_n_past_choices = [0, 1, 5]
                                    recall_stride_choices = [1, 3]

                                for recall_stride in recall_stride_choices:
                                    for recall_n_past in recall_n_past_choices:

                                        recall_n_future_choices = []
                                        if len(short_time_recall) > 0 and recall_n_past > 0:
                                            recall_n_future_choices = [0, 1, 5]
                                        elif len(short_time_recall) > 0:
                                            recall_n_future_choices = [1, 5]

                                        for recall_n_future in recall_n_future_choices:
                                            for positional_encoder_type in ["alibi"]:
                                                for block_configuration in [[Dimensions.NEXT_MEASUREMENT, Dimensions.NEXT_MEASUREMENT],
                                                                            [Dimensions.NEXT_MEASUREMENT, Dimensions.STATE_INPUT]]:
                                                    for state_activation in ["tanh", "ltanh", None]:
                                                        state_regularizer = None
                                                        if state_activation is None:
                                                            state_regularizer = "mse"

                                                        n_variation += 1

                                                        config = {"n_segment": n_segment,
                                                                  "fast_forward": fast_forward,
                                                                  "stable_state_epochs": stable_state_epochs,
                                                                  "check_input_masks": check_input_masks,
                                                                  "state_save_method": state_save_method,
                                                                  "mask_sensorial_data": mask_sensorial_data,
                                                                  "short_time_recall": short_time_recall,
                                                                  "recall_stride_past": recall_stride,
                                                                  "recall_stride_future": recall_stride,
                                                                  "recall_n_past": recall_n_past,
                                                                  "recall_n_future": recall_n_future,
                                                                  "positional_encoder_type": positional_encoder_type,
                                                                  "block_configuration": block_configuration,
                                                                  "state_activation": state_activation}

                                                        model = make_model(
                                                            config, "ParametersModel").model_validate(config)
                                                        model_json = model.model_dump_json()
                                                        variation_hash = hash(
                                                            model_json)

                                                        configurations[variation_hash] = config

    assert len(configurations) == n_variation
    configurations_path = os.path.join(output_dir, "configurations.bin")
    with open(configurations_path, "wb") as file:
        pickle.dump(configurations, file)

    toy1d_parameter_variation = configurations

    aditional_outputs = ["save_toy1d_metrics",
                         "save_toy1d_metrics_sample_logits",
                         "save_toy1d_metrics_sample_plots",
                         "save_toy1d_autoregressive_metrics"]

    output = d_parameter_variation.execute(["save_toy1d_parameter_variation_plots"],
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
