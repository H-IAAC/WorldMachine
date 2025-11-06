import glob
import json
import os

import numpy as np
import pandas as pd
from hamilton.function_modifiers import datasaver

exclude_paths = ["final_results"]


def encoder(obj):
    '''
        Encode a Data class to JSON

        Implements encoding of numpy data types. Other data types are returned as dicts

        Parameters:
            obj (Object): object to be encoded

        Returns
            @returns encoded object
    '''

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)

    return obj.__dict__


def decoder(obj: dict):
    for key in obj:
        if isinstance(obj[key], dict):
            obj[key] = decoder(obj[key])
        elif isinstance(obj[key], list):
            obj[key] = np.array(obj[key])

    return obj


@datasaver()
def save_metrics(metrics: dict | pd.DataFrame, output_dir: str, metrics_name: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, metrics_name+".json")

    with open(file_path, "w", encoding="utf-8") as file:
        if isinstance(metrics, pd.DataFrame):
            file.write(metrics.T.to_json())
        else:
            json.dump(metrics, file, default=encoder)

    return {"path": file_path}


def load_metrics(output_dir: str, metrics_name: str) -> dict:
    file_path = os.path.join(output_dir, metrics_name+".json")
    with open(file_path, "r", encoding="utf-8") as file:
        metrics = json.load(file, object_hook=decoder)

    return metrics


def load_multiple_metrics(output_dir: str, metrics_name: str) -> dict[str, dict]:
    metrics = {}

    paths = glob.glob(os.path.join(output_dir, "*"))
    for path in paths:
        if os.path.isdir(path):
            experiment_name = os.path.basename(path)

            if experiment_name not in exclude_paths:
                file_path = os.path.join(path, metrics_name+".json")

                with open(file_path, "r", encoding="utf-8") as file:
                    metrics[experiment_name] = json.load(
                        file, object_hook=decoder)

    return metrics


def get_values(metrics, indexes, to_numpy: bool = True):
    values = []

    for name in metrics:
        value = metrics[name]

        for index in indexes:
            value = value[index]

        values.append(value)

    if to_numpy:
        values = np.array(values)

    return values
