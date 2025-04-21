import torch
from hamilton import driver
from hamilton_sdk import adapters
from torch.optim import AdamW
from world_machine_experiments import shared
from world_machine_experiments.toy1d import Dimensions, parameter_variation

from world_machine.train.scheduler import ChoiceScheduler, UniformScheduler

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

    if long:
        n_epoch = 20
        output_dir = "toy1d_memory_long2"
    else:
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
        # "NoDiscover_M0-90": {"discover_state": False, "mask_sensorial_data": UniformScheduler(0, 0.9, n_epoch)},
        # "Break1": {"discover_state": True, "n_segment": 2},
        # "Break1_M0-100": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch)},
        # "2Break1_M0-100_FF": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True},
        # "Break4_M0-100_FF": {"discover_state": True, "n_segment": 5, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True},
        # "MLP_Break1_M0-100_FF_TRAINM": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT]},
        # "MLP_Break1_M0-100_FF_TRAINM_SS12": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT], "state_size": 12},
        # "MLP_Break1_M0-100_FF_LTANH": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "state_activation": "ltanh"},
        # "MLP_Break1_M0or100_FF": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": ChoiceScheduler([0, 1], n_epoch), "fast_forward": True},
        # "Break1_M0-100_FF_Alibi": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "positional_encoder_type": "alibi"},
        # "Break1_M0-100_FF_H01": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_size": 1, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT]},
        # "Break1_M0-100_FF_Hx1": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_size": 1, "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT]},
        # "Break1_M0-100_FF_H0x": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT]},
        # "Break1_M0-100_FF_TRAINMC_H0x": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT, Dimensions.STATE_CONTROL]},
        # "SMSM_Break1_M0-100_FF_TRAINMC_SS12_H0x": {"block_configuration": [Dimensions.STATE, Dimensions.NEXT_MEASUREMENT, Dimensions.STATE, Dimensions.NEXT_MEASUREMENT], "state_size": 12, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT, Dimensions.STATE_CONTROL]},
        # "SMSM_Break1_M0-100_FF_TRAINM_SS12_H0x": {"block_configuration": [Dimensions.STATE, Dimensions.NEXT_MEASUREMENT, Dimensions.STATE, Dimensions.NEXT_MEASUREMENT], "state_size": 12, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT]},
        # "SMSM_Break1_M0-100_FF_Alibi_TRAINM_SS12_H0x": {"block_configuration": [Dimensions.STATE, Dimensions.NEXT_MEASUREMENT, Dimensions.STATE, Dimensions.NEXT_MEASUREMENT], "state_size": 12, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT], "positional_encoder_type": "alibi"},
        # "Break1_M0-100_FF_3H_SS12": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "n_attention_head": 3, "state_size": 12},
        # "SM_Break1_M0-100_FF_3H_SS12_TrainM_H0x": {"block_configuration": [Dimensions.STATE, Dimensions.NEXT_MEASUREMENT], "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT], "n_attention_head": 3, "state_size": 12},
        # "Break1_M0-100_FF_H0x_SS12_TrainM_STR-MD": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT], "state_size": 12, "short_time_recall": set([Dimensions.NEXT_MEASUREMENT, Dimensions.STATE_DECODED])},
        # "Break1_M0-100_FF_H0x_SS12_TrainM_STR5-MD": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT], "state_size": 12, "short_time_recall": set([Dimensions.NEXT_MEASUREMENT, Dimensions.STATE_DECODED]), "recall_n_past": 5},
        # "Break1_M0-100_FF_H0x_SS12_TrainM_STR10-MD": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT], "state_size": 12, "short_time_recall": set([Dimensions.NEXT_MEASUREMENT, Dimensions.STATE_DECODED]), "recall_n_past": 10},
        # "M0-100_FF_H0x_SS12_TrainM_STR5-MD": {"discover_state": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT], "state_size": 12, "short_time_recall": set([Dimensions.NEXT_MEASUREMENT, Dimensions.STATE_DECODED]), "recall_n_past": 5},
        # "Break1_M0-100_FF_H0x_SS12_TrainM_STR-MD_StateReg": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT], "state_size": 12, "short_time_recall": set([Dimensions.NEXT_MEASUREMENT, Dimensions.STATE_DECODED]), "state_regularizer": "mse", "state_activation": None},
        # "Break1_M0-100_StateReg": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "state_regularizer": "mse", "state_activation": None},
        # "Break1_M0-100_FF_H0x_SS12_TrainM_STR5-MD_StateReg":  {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT], "state_size": 12, "short_time_recall": set([Dimensions.NEXT_MEASUREMENT, Dimensions.STATE_DECODED]), "state_regularizer": "mse", "state_activation": None,  "recall_n_past": 5},
        "Break1_M0-100_FF_H0x_SS4_TrainM_STR5-MD_StateReg":  {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.NEXT_MEASUREMENT], "state_size": 4, "short_time_recall": set([Dimensions.NEXT_MEASUREMENT, Dimensions.STATE_DECODED]), "state_regularizer": "mse", "state_activation": None,  "recall_n_past": 5},
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
                                                   "custom_plots": {"StateReg": ["Break1_M0-100", "Break1_M0-100_FF_H0x_SS12_TrainM_STR-MD_StateReg", "Break1_M0-100_StateReg", "Break1_M0-100_FF_H0x_SS12_TrainM_STR5-MD_StateReg", "Break1_M0-100_FF_H0x_SS4_TrainM_STR5-MD_StateReg"], },
                                                   # "aditional_outputs": aditional_outputs
                                                   },
                                           # overrides={
                                           #    "base_dir": output_dir}
                                           )
