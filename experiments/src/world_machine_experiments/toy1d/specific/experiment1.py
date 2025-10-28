import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import networkx as nx

from world_machine_experiments.shared.save_metrics import load_multiple_metrics, get_values
from world_machine_experiments.toy1d.dimensions import Dimensions
from world_machine.train.stages import StateSaveMethod
from world_machine_experiments.shared.acronyms import format_name
from world_machine_experiments.shared import function_variation
from hamilton.function_modifiers import source, value
import os
from hamilton.function_modifiers import datasaver

plt.rcParams["mathtext.default"] = "regular"

# Default Data
color_map = {
    "Base": "#60BF60",
    "SensorialMasker": "#D36767",
    "CompleteProtocol": "#6060BF"
}

palette = list(color_map.values())

metric_names = ["normal", "use_state", "prediction",
                "prediction_shallow", "prediction_local"]


# Load variations data

metrics = function_variation({"output_dir": source("data_dir"),
                              "metrics_name": value("toy1d_metrics")},
                             "metrics")(load_multiple_metrics)

parameters = function_variation({"output_dir": source("data_dir"),
                                 "metrics_name": value("parameters")},
                                "parameters")(load_multiple_metrics)

train_history = function_variation({"output_dir": source("data_dir"),
                                    "metrics_name": value("train_history")},
                                   "train_history")(load_multiple_metrics)


def variations_df(metrics: dict[str, dict],
                  parameters: dict[str, dict],
                  train_history: dict[str, dict]) -> pd.DataFrame:
    variations_data = []

    for name in parameters:
        item = parameters[name]["parameters"]
        item["name"] = name

        for m in metric_names:
            m_1 = m
            m_2 = m

            for criterion in ["mse", "0.1sdtw"]:
                item[f"{m_1}_{criterion}"] = metrics[name]["means"][m_2][f"state_decoded_{criterion}"]

        item["duration"] = train_history[name]["means"]["duration"].sum()
        item["parameters"] = parameters[name]

        variations_data.append(item)

    df = pd.DataFrame(variations_data)

    mask = np.ones(len(df), bool)
    for metric in metric_names:

        mask = np.bitwise_and(mask, np.bitwise_not(df[metric+"_mse"].isna()))

        data = df[f"{metric}_mse"].to_numpy()

        data_mask = np.bitwise_not(np.isnan(data))
        data_mask = np.bitwise_and(data_mask, data < 1)

        data_max = data[data_mask].mean() + 3*data[data_mask].std()

        mask = np.bitwise_and(mask, np.bitwise_not(np.isnan(data)))
        mask = np.bitwise_and(mask, data < data_max)

    df["mask"] = mask

    return df


def masked_percentage(variations_df: pd.DataFrame) -> float:
    return variations_df["mask"].sum()/variations_df


@datasaver()
def save_masked_percentage(masked_percentage: float, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, metrics_name+".json")

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, default=encoder)

    return {"path": file_path}
