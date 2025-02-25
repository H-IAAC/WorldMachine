import glob
import os

import numpy as np


def load_train_history(base_dir: str, history_file_name: str) -> dict[str, dict[str, np.ndarray]]:
    train_history = {}

    paths = glob.glob(os.path.join(base_dir, "*"))
    for path in paths:
        if os.path.isdir(path):
            experiment_name = os.path.basename(path)
            history_path = os.path.join(path, history_file_name+".npz")

            train_history[experiment_name] = dict(np.load(history_path))

    return train_history


