import multiprocessing as mp

import torch
from hamilton import driver
from hamilton_sdk import adapters
from torch.optim import SGD, AdamW
from world_machine_experiments import shared
from world_machine_experiments.toy1d import Channels, parameter_variation

from world_machine.train.scheduler import ChoiceScheduler, UniformScheduler

if __name__ == "__main__":
    # tracker = adapters.HamiltonTracker(
    #    project_id=1,
    #    username="EltonCN",
    #    dag_name="toy1d_parameter_variation"
    # )

    mp.set_start_method("spawn")

    d_parameter_variation = driver.Builder().with_modules(
        parameter_variation, shared).build()  # .with_adapter(tracker).build()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    long = False
    long2 = False
    long3 = True

    if long3:
        n_epoch = 100
        output_dir = "toy1d_memory_long4"
    elif long2:
        n_epoch = 15
        output_dir = "toy1d_memory_long3"
    elif long:
        n_epoch = 20
        output_dir = "toy1d_memory_long2"
    else:
        n_epoch = 5
        output_dir = "toy1d_memory"

    toy1d_base_args = {"sequence_length": 1000,
                       "n_sequence": 10000,
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
        # "Base": {"discover_state": True},
        # "M0-90": {"discover_state": True, "mask_sensorial_data": UniformScheduler(0, 0.9, n_epoch)},
        # "NoDiscover_M0-90": {"discover_state": False, "mask_sensorial_data": UniformScheduler(0, 0.9, n_epoch)},
        # "Break1": {"discover_state": True, "n_segment": 2},
        # "Break1_M0-100": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch)},
        # "2Break1_M0-100_FF": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True},
        # "Break4_M0-100_FF": {"discover_state": True, "n_segment": 5, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True},
        # "MLP_Break1_M0-100_FF_TRAINM": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "sensorial_train_losses": [Dimensions.measurement]},
        # "MLP_Break1_M0-100_FF_TRAINM_SS12": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "sensorial_train_losses": [Dimensions.measurement], "state_size": 12},
        # "MLP_Break1_M0-100_FF_LTANH": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "state_activation": "ltanh"},
        # "MLP_Break1_M0or100_FF": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": ChoiceScheduler([0, 1], n_epoch), "fast_forward": True},
        # "Break1_M0-100_FF_Alibi": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "positional_encoder_type": "alibi"},
        # "Break1_M0-100_FF_H01": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_size": 1, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_Hx1": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_size": 1, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_TRAINMC_H0x": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement, Dimensions.STATE_CONTROL]},
        # "SMSM_Break1_M0-100_FF_TRAINMC_SS12_H0x": {"block_configuration": [Dimensions.STATE, Dimensions.measurement, Dimensions.STATE, Dimensions.measurement], "state_size": 12, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement, Dimensions.STATE_CONTROL]},
        # "SMSM_Break1_M0-100_FF_TRAINM_SS12_H0x": {"block_configuration": [Dimensions.STATE, Dimensions.measurement, Dimensions.STATE, Dimensions.measurement], "state_size": 12, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "SMSM_Break1_M0-100_FF_Alibi_TRAINM_SS12_H0x": {"block_configuration": [Dimensions.STATE, Dimensions.measurement, Dimensions.STATE, Dimensions.measurement], "state_size": 12, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement], "positional_encoder_type": "alibi"},
        # "Break1_M0-100_FF_3H_SS12": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "n_attention_head": 3, "state_size": 12},
        # "SM_Break1_M0-100_FF_3H_SS12_TrainM_H0x": {"block_configuration": [Dimensions.STATE, Dimensions.measurement], "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement], "n_attention_head": 3, "state_size": 12},
        # "Break1_M0-100_FF_H0x_SS12_TrainM_STR-MD": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement], "state_size": 12, "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED])},
        # "Break1_M0-100_FF_H0x_SS12_TrainM_STR5-MD": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement], "state_size": 12, "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5},
        # "Break1_M0-100_FF_H0x_SS12_TrainM_STR10-MD": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement], "state_size": 12, "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 10},
        # "M0-100_FF_H0x_SS12_TrainM_STR5-MD": {"discover_state": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement], "state_size": 12, "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5},
        # "Break1_M0-100_FF_H0x_SS12_TrainM_STR-MD_StateReg": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement], "state_size": 12, "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "state_regularizer": "mse", "state_activation": None},
        # "Break1_M0-100_StateReg": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "state_regularizer": "mse", "state_activation": None},
        # "Break1_M0-100_FF_H0x_SS12_TrainM_STR5-MD_StateReg":  {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement], "state_size": 12, "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "state_regularizer": "mse", "state_activation": None,  "recall_n_past": 5},
        # "Break1_M0-100_FF_H0x_SS4_TrainM_STR5-MD_StateReg":  {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement], "state_size": 4, "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "state_regularizer": "mse", "state_activation": None,  "recall_n_past": 5},
        # "Break1_M0-100_FF_H0x_SS12_TrainM_STR5W-MD": {"discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement], "state_size": 12, "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5},
        # "Break1_M0-100_FF_H0x_SS12_TrainM_STR5W-MD_SGD": {"learning_rate": 1e-2, "optimizer_class": SGD, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement], "state_size": 12, "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5},
        # "Break1_M0-100_FF_H0x_SS12_TrainM_STR5W-MD_SGD2": {"learning_rate": 1e-1, "optimizer_class": SGD, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement], "state_size": 12, "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5},
        # "Break1_M0-100_FF_H0x_SS12_TrainM_STR5W-MD_SGD3": {"learning_rate": 1e-3, "optimizer_class": SGD, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement], "state_size": 12, "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5},
        # "Break8_M0-100_FF_H0x_SS12_TrainM_STR5-MD": {"discover_state": True, "n_segment": 9, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement], "state_size": 12, "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5},
        # "Break1_M0-100_FF_H0x_SCheckSensorial": {"check_input_masks": True, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_Alibi": {"positional_encoder_type": "alibi", "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_Sine": {"positional_encoder_type": "sine", "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_Alibi_SCR01": {"state_cov_regularizer": 0.1, "positional_encoder_type": "alibi", "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_Alibi_SCR001": {"state_cov_regularizer": 0.01, "positional_encoder_type": "alibi", "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_Alibi_SCR1e-3": {"state_cov_regularizer": 1e-3, "positional_encoder_type": "alibi", "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_Alibi_SCR1e-4": {"state_cov_regularizer": 1e-4, "positional_encoder_type": "alibi", "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_Alibi_SCR1e-5": {"state_cov_regularizer": 1e-5, "positional_encoder_type": "alibi", "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi": {"positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS512_Alibi": {"positional_encoder_type": "alibi", "state_size": 512, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial": {"check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS512v_SCheckSensorial": {"check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 512, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head2": {"n_attention_head": 2, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4": {"n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head8": {"n_attention_head": 8, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_STR5-MD_StateReg": {"state_regularizer": "mse", "state_activation": None, "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5, "n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_StateReg": {"state_regularizer": "mse", "state_activation": None, "n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break2_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4": {"n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 3, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break2_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_STR5-MD": {"short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5, "n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 3, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_SCheckSensorial_Head4_STR5-MD": {"short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5, "n_attention_head": 4, "check_input_masks": True, "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_H0x_SS128_Alibi_SCheckSensorial_Head4_STR5-MD": {"short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5, "n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": False, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "M0-100_H0x_SS128_Alibi_SCheckSensorial_Head4": {"n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS256_Alibi_SCheckSensorial_Head4": {"n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 256, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_SDroput01": {"state_dropout": 0.1, "n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4001": {"state_dropout": 0.01, "n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head402": {"state_dropout": 0.2, "n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_SinTanh": {"state_activation": "sintanh", "n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_Sin": {"state_activation": "sin", "n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_SCR1e-2": {"state_cov_regularizer": 1e-2, "n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "H0x_SS128_Alibi_Head4": {"n_attention_head": 4, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        #
        # "Break20_M0-100_FF_H0x_SS128_Alibi_Head4": {"mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "n_segment": 21, "n_attention_head": 4, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break40_M0-100_FF_H0x_SS128_Alibi_Head4": {"mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "n_segment": 41, "n_attention_head": 4, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break20_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4": {"check_input_masks": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "n_segment": 21, "n_attention_head": 4, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},


        # "M0-100_H0x_SS128_Alibi_Head4": {"n_attention_head": 4, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "M0-100_H0x_SS128_Alibi_SCheckSensorial_Head4": {"n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_H0x_SS128_Alibi_SCheckSensorial_Head4": {"n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4": {"n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_STR5-MD": {"short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5, "n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_H0x_SS128_Alibi_Head4": {"n_attention_head": 4, "check_input_masks": False, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_Head4": {"n_attention_head": 4, "check_input_masks": False, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_Head4_STR5-MD": {"short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5, "n_attention_head": 4, "check_input_masks": False, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "M0-100_H0x_SS128_Alibi_Head4_MMMM": {"block_configuration": [Dimensions.measurement, Dimensions.measurement, Dimensions.measurement, Dimensions.measurement], "n_attention_head": 4, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "M0-100_H0x_SS256_Alibi_Head4_MMMM": {"block_configuration": [Dimensions.measurement, Dimensions.measurement, Dimensions.measurement, Dimensions.measurement], "n_attention_head": 4, "positional_encoder_type": "alibi", "state_size": 256, "discover_state": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "M0-100_H0x_SS128_Alibi_Head4_SMSM": {"block_configuration": [Dimensions.STATE, Dimensions.measurement, Dimensions.STATE, Dimensions.measurement], "n_attention_head": 4, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_Head4_STR5F-MD": {"short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_future": 5, "n_attention_head": 4, "check_input_masks": False, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_STR5F-MD": {"short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_future": 5, "n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_Head4_STR5F5P-MD": {"short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5, "recall_n_future": 5, "n_attention_head": 4, "check_input_masks": False, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_STR5F5P-MD": {"short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5, "recall_n_future": 5, "n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS256_Alibi_SCheckSensorial_Head8_MMM_STR5F5P-MD": {"block_configuration": [Dimensions.measurement, Dimensions.measurement], "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5, "recall_n_future": 5, "n_attention_head": 8, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 256, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS256_Alibi_SCheckSensorial_Head8_MMM_STR5F5PStride3F3P-MD": {"recall_stride_past": 3, "recall_stride_future": 3, "block_configuration": [Dimensions.measurement, Dimensions.measurement], "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5, "recall_n_future": 5, "n_attention_head": 8, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 256, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS256_Alibi_SCheckSensorial_Head8_MMM_STR5F5PStride6F6P-MD": {"recall_stride_past": 6, "recall_stride_future": 6, "block_configuration": [Dimensions.measurement, Dimensions.measurement], "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5, "recall_n_future": 5, "n_attention_head": 8, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 256, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_STR5F5PStride3F3P-MD": {"recall_stride_past": 3, "recall_stride_future": 3, "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5, "recall_n_future": 5, "n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "alibi", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},

        # "H0x_SS128_Sine_Head4": {"n_attention_head": 4, "positional_encoder_type": "sine", "state_size": 128, "discover_state": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        # "Break1_M0-100_FF_H0x_SS128_Sine_SCheckSensorial_Head4_STR5F5PStride3F3P-MD": {"recall_stride_past": 3, "recall_stride_future": 3, "short_time_recall": set([Dimensions.measurement, Dimensions.STATE_DECODED]), "recall_n_past": 5, "recall_n_future": 5, "n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "sine", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Dimensions.measurement]},
        "TESTE": {"recall_stride_past": 3, "recall_stride_future": 3, "short_time_recall": set([Channels.MEASUREMENT, Channels.STATE_DECODED]), "recall_n_past": 5, "recall_n_future": 5, "n_attention_head": 4, "check_input_masks": True, "positional_encoder_type": "sine", "state_size": 128, "discover_state": True, "n_segment": 2, "mask_sensorial_data": UniformScheduler(0, 1, n_epoch), "fast_forward": True, "measurement_shift": 0, "sensorial_train_losses": [Channels.MEASUREMENT]},
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
                                                   "custom_plots": {  # "BigState": ["Break1_M0-100_FF_H0x_SS128_Alibi", "Break1_M0-100_FF_H0x_SS512_Alibi", "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial", "Break1_M0-100_FF_H0x_SS512v_SCheckSensorial", "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head2", "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4", "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head8", "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_STR5-MD", "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_STR5-MD_StateReg", "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_StateReg", "Break2_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4", "Break2_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_STR5-MD",
                                                       #             "Break2_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_STR5-MD",
                                                       #             "Break1_M0-100_FF_H0x_SS128_SCheckSensorial_Head4_STR5-MD",
                                                       #             "Break1_M0-100_H0x_SS128_Alibi_SCheckSensorial_Head4_STR5-MD",
                                                       #             "M0-100_H0x_SS128_Alibi_SCheckSensorial_Head4",
                                                       #             "Break1_M0-100_FF_H0x_SS256_Alibi_SCheckSensorial_Head4",
                                                       #             "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_SDroput01",
                                                       #             "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4001",
                                                       #             "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head402",
                                                       #             "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_SinTanh",
                                                       #             "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_Sin",
                                                       #             "Break1_M0-100_FF_H0x_SS128_Alibi_SCheckSensorial_Head4_SCR1e-2"
                                                       #             ],
                                                       # "StateReg": ["Break1_M0-100", "Break1_M0-100_FF_H0x_SS12_TrainM_STR-MD_StateReg", "Break1_M0-100_StateReg", "Break1_M0-100_FF_H0x_SS12_TrainM_STR5-MD_StateReg", "Break1_M0-100_FF_H0x_SS4_TrainM_STR5-MD_StateReg"],
                                                       # "SGD": ["Break1_M0-100_FF_H0x_SS12_TrainM_STR5W-MD_SGD", "Break1_M0-100_FF_H0x_SS12_TrainM_STR5W-MD_SGD2", "Break1_M0-100_FF_H0x_SS12_TrainM_STR5W-MD_SGD3"],
                                                       # "CheckSensorial": ["Break1_M0-100_FF_H0x", "Break1_M0-100_FF_H0x_SCheckSensorial",  "Break1_M0-100_FF_H0x_Alibi", "Break1_M0-100_FF_H0x_Sine"],
                                                       # "StateCovReg": ["Break1_M0-100_FF_H0x_Alibi", "Break1_M0-100_FF_H0x_Alibi_SCR01", "Break1_M0-100_FF_H0x_Alibi_SCR001", "Break1_M0-100_FF_H0x_Alibi_SCR1e-3", "Break1_M0-100_FF_H0x_Alibi_SCR1e-4", "Break1_M0-100_FF_H0x_Alibi_SCR1e-5"],
                                                   },
                                                   # "aditional_outputs": aditional_outputs
                                                   },
                                           # overrides={
                                           #    "base_dir": output_dir}
                                           )
