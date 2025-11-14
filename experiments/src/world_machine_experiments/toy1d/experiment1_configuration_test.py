import hashlib
import os
import pickle
import random

import torch
import torch.multiprocessing as mp
from hamilton import driver
from torch.optim import AdamW

from world_machine.train.scheduler import UniformScheduler
from world_machine.train.stages import StateSaveMethod
from world_machine_experiments import shared
from world_machine_experiments.shared.pipeline import save_pipeline
from world_machine_experiments.shared.save_parameters import make_model
from world_machine_experiments.toy1d import Channels, parameter_variation
from world_machine_experiments.toy1d.specific import experiment1

if __name__ == "__main__":
    output_dir = "toy1d_experiment1_configuration_test"

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

    n_epoch = 100

    toy1d_base_args = {"sequence_length": 1000,
                       "n_sequence": 10000,
                       "context_size": 200,
                       "batch_size": 256,
                       "n_epoch": n_epoch,
                       "accumulation_steps": 1,
                       "state_dimensions": [0],
                       "device": devices,
                       "state_control": "periodic",
                       "discover_state": True,
                       "sensory_train_losses": [Channels.MEASUREMENT],
                       }

    n_variation = 0
    configurations = {}
    for n_segment in [2, 1]:
        fast_forward_choices = [False]
        if n_segment > 1:
            fast_forward_choices = [True, False]

        for fast_forward in fast_forward_choices:
            for stable_state_epochs in [1]:
                for check_input_masks in [True, False]:
                    for state_save_method in [StateSaveMethod.MEAN, StateSaveMethod.REPLACE]:
                        for mask_sensory_data in [UniformScheduler(0, 1, n_epoch)]:
                            for short_time_recall in [{Channels.MEASUREMENT}, set()]:

                                recall_n_past_choices = [0]
                                if len(short_time_recall) > 0:
                                    recall_n_past_choices = [5, 1, 0]

                                for recall_n_past in recall_n_past_choices:

                                    recall_n_future_choices = [0]
                                    if len(short_time_recall) > 0 and recall_n_past > 0:
                                        recall_n_future_choices = [5, 1, 0]
                                    elif len(short_time_recall) > 0:
                                        recall_n_future_choices = [5, 1]

                                    for recall_n_future in recall_n_future_choices:

                                        recall_stride_past_choices = [0]
                                        if recall_n_past > 0:
                                            recall_stride_past_choices = [3, 1]

                                        recall_stride_future_choices = [0]
                                        if recall_n_future > 0:
                                            recall_stride_future_choices = [
                                                3, 1]

                                        for recall_stride_past in recall_stride_past_choices:
                                            for recall_stride_future in recall_stride_future_choices:
                                                for positional_encoder_type in ["alibi"]:
                                                    for block_configuration in [[Channels.MEASUREMENT, Channels.MEASUREMENT],
                                                                                [Channels.MEASUREMENT, Channels.STATE_INPUT]]:
                                                        for state_activation in ["tanh", None]:
                                                            state_regularizer = None
                                                            if state_activation is None:
                                                                state_regularizer = "mse"

                                                            for train_mse in [True]:
                                                                for train_stdw in [False]:
                                                                    for noise_config in [None,
                                                                                         {"state": {
                                                                                             "mean": 0.0, "std": 0.1}},
                                                                                         {"measurement": {
                                                                                             "mean": 0.0, "std": 0.1}},
                                                                                         {"state": {"mean": 0.0, "std": 0.1}, "measurement": {"mean": 0.0, "std": 0.1}}]:
                                                                        for local_chance in [None, 0.25]:
                                                                            n_variation += 1

                                                                            config = {"n_segment": n_segment,
                                                                                      "fast_forward": fast_forward,
                                                                                      "stable_state_epochs": stable_state_epochs,
                                                                                      "check_input_masks": check_input_masks,
                                                                                      "state_save_method": state_save_method,
                                                                                      "mask_sensory_data": mask_sensory_data,
                                                                                      "short_time_recall": short_time_recall,
                                                                                      "recall_stride_past": recall_stride_past,
                                                                                      "recall_stride_future": recall_stride_future,
                                                                                      "recall_n_past": recall_n_past,
                                                                                      "recall_n_future": recall_n_future,
                                                                                      "positional_encoder_type": positional_encoder_type,
                                                                                      "block_configuration": block_configuration,
                                                                                      "state_activation": state_activation,
                                                                                      "state_regularizer": state_regularizer,
                                                                                      "train_mse": train_mse,
                                                                                      "train_sdtw": train_stdw,
                                                                                      "noise_config": noise_config,
                                                                                      "local_chance": local_chance,

                                                                                      # Fixed
                                                                                      "batch_size": 256,
                                                                                      "learning_rate": 1e-3,
                                                                                      "cosine_annealing": True,
                                                                                      "cosine_annealing_T_mult": 1,
                                                                                      "cosine_annealing_T0": 25,
                                                                                      "weight_decay": 5e-5,
                                                                                      "optimizer_class": AdamW,
                                                                                      "state_size": 128,
                                                                                      "n_attention_head": 4}

                                                                            model = make_model(
                                                                                config, "ParametersModel").model_validate(config)
                                                                            model_json = model.model_dump_json()
                                                                            variation_hash = int(
                                                                                hashlib.md5(model_json.encode('utf-8')).hexdigest(), 16)

                                                                            configurations["variation"+str(
                                                                                variation_hash)] = config

    items = list(configurations.items())
    random.shuffle(items)
    configurations = dict(items)

    assert len(configurations) == n_variation

    print(f"Running {n_variation} variations")

    os.makedirs(output_dir, exist_ok=True)
    configurations_path = os.path.join(output_dir, "configurations.bin")
    with open(configurations_path, "wb") as file:
        pickle.dump(configurations, file)

    toy1d_parameter_variation = configurations

    aditional_outputs = ["save_toy1d_metrics"]

    final_vars = ["save_toy1d_parameter_variation_info"]
    save_pipeline(d_parameter_variation, final_vars,
                  "model_train_pipeline", output_dir)
    try:
        output = d_parameter_variation.execute(final_vars,
                                               inputs={"base_seed": 42,
                                                       "output_dir": output_dir,
                                                       "n_run": 1,
                                                       "toy1d_base_args": toy1d_base_args,
                                                       "n_worker": n_worker,
                                                       "max_jobs_per_device": max_jobs_per_device,
                                                       "toy1d_parameter_variation": toy1d_parameter_variation,
                                                       "aditional_outputs": aditional_outputs,
                                                       "minimal": True
                                                       }
                                               )
    except Exception as e:
        print("ERROR")
        raise e

    print("Running Analysis")

    d_experiment1 = driver.Builder().with_modules(experiment1, shared).build()

    final_vars = ["save_masked_percentage",
                  "save_task_distribution_plots",
                  "save_tasks_correlation",
                  "save_tasks_correlation_plots",
                  "save_divergence_probability",
                  "save_filtered_divergence_probability",
                  "save_divergence_probability_plots",
                  "save_filtered_divergence_probability_plots",
                  "save_impact_test_df",
                  "save_impact_test_full_df",
                  "save_joint_impact",
                  "save_joint_impact_plots",
                  "save_filtered_marginal_impact",
                  "save_filtered_marginal_impact_plots",
                  "save_task_impact_plots",
                  "save_duration_impact_plots",
                  "save_best_configurations",
                  "save_best_models",
                  "save_best_models_metrics_table",
                  ]
    save_pipeline(d_experiment1, final_vars,
                  "experiment_pipeline", output_dir)

    d_experiment1.execute(final_vars,

                          inputs={"data_dir": output_dir,
                                  "output_dir": os.path.join(output_dir, "final_results")})

    print("END")
