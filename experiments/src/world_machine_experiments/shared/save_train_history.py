import os

import numpy as np
from hamilton.function_modifiers import datasaver


@datasaver()
def save_train_history(train_history: dict[str, np.ndarray], output_dir: str, history_name: str = "train_history") -> dict:
    path = os.path.join(output_dir, history_name)

    np.savez_compressed(path, **train_history)

    info = {"history_path": path+".npz"}

    return info
