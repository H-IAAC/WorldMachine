
import numpy as np
import pandas as pd
from hamilton.function_modifiers import source, value

from world_machine.train.stages import StateSaveMethod
from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.save_metrics import load_multiple_metrics
from world_machine_experiments.toy1d.channels import Channels

from ._shared import *

metrics = function_variation({"output_dir": source("data_dir"),
                              "metrics_name": value("toy1d_metrics")},
                             "metrics")(load_multiple_metrics)

parameters = function_variation({"output_dir": source("data_dir"),
                                 "metrics_name": value("parameters")},
                                "parameters")(load_multiple_metrics)

train_history = function_variation({"output_dir": source("data_dir"),
                                    "metrics_name": value("toy1d_train_history")},
                                   "train_history")(load_multiple_metrics)


def _add_diverge_mask(df: pd.DataFrame) -> None:
    mask = np.ones(len(df), bool)
    for metric in task_names:

        mask = np.bitwise_and(mask, np.bitwise_not(df[metric+"_mse"].isna()))

        data = df[f"{metric}_mse"].to_numpy()

        data_mask = np.bitwise_not(np.isnan(data))
        data_mask = np.bitwise_and(data_mask, data < 1)

        data_max = data[data_mask].mean() + 3*data[data_mask].std()

        mask = np.bitwise_and(mask, np.bitwise_not(np.isnan(data)))
        mask = np.bitwise_and(mask, data < data_max)

    df["mask"] = mask


def _add_variables(df: pd.DataFrame) -> None:
    df["SB_1"] = np.bitwise_and(df["n_segment"] == 2, df["fast_forward"] == False)  # nopep8
    df["SB_2"] = np.bitwise_and(df["n_segment"] == 2, df["fast_forward"] == True)  # nopep8

    df["SM_1"] = df["state_save_method"] == StateSaveMethod.MEAN.value  # nopep8
    df["SM_2"] = df["check_input_masks"] == True  # nopep8

    df["AC_1"] = pd.isnull(df["state_activation"])  # nopep8

    df["MD_1"] = df["block_configuration"].map(lambda x: np.all(x == [Channels.MEASUREMENT.value, Channels.STATE_INPUT.value]))  # nopep8

    df["NA_1"] = df["noise_config"].map(lambda x: x is not None and "state" in x)  # nopep8
    df["NA_2"] = df["noise_config"].map(lambda x: x is not None and "measurement" in x)  # nopep8

    df["RP_1"] = np.bitwise_and(df["recall_stride_past"] == 1, df["recall_n_past"] == 1)  # nopep8
    df["RP_2"] = np.bitwise_and(df["recall_stride_past"] == 1, df["recall_n_past"] == 5)  # nopep8
    df["RP_3"] = np.bitwise_and(df["recall_stride_past"] == 3, df["recall_n_past"] == 1)  # nopep8
    df["RP_4"] = np.bitwise_and(df["recall_stride_past"] == 3, df["recall_n_past"] == 5)  # nopep8

    df["RF_1"] = np.bitwise_and(df["recall_stride_future"] == 1, df["recall_n_future"] == 1)  # nopep8
    df["RF_2"] = np.bitwise_and(df["recall_stride_future"] == 1, df["recall_n_future"] == 5)  # nopep8
    df["RF_3"] = np.bitwise_and(df["recall_stride_future"] == 3, df["recall_n_future"] == 1)  # nopep8
    df["RF_4"] = np.bitwise_and(df["recall_stride_future"] == 3, df["recall_n_future"] == 5)  # nopep8

    df["LM_1"] = np.bitwise_not(pd.isnull(df["local_chance"]))  # nopep8


def variations_df(metrics: dict[str, dict],
                  parameters: dict[str, dict],
                  train_history: dict[str, dict]) -> pd.DataFrame:
    variations_data = []

    for name in parameters:
        item = parameters[name]["parameters"]
        item["name"] = name

        for m in task_names:
            m_1 = m
            m_2 = m

            for criterion in ["mse", "0.1sdtw"]:
                item[f"{m_1}_{criterion}"] = metrics[name]["means"][m_2][f"state_decoded_{criterion}"]

        item["duration"] = train_history[name]["means"]["duration"].sum()

        variations_data.append(item)

    df = pd.DataFrame(variations_data)

    _add_diverge_mask(df)
    _add_variables(df)

    return df
