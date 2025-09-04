import numpy as np
from hamilton.function_modifiers import source, value

from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.save_metrics import save_metrics


def toy1d_autoregressive_metrics(toy1d_metrics: dict[str, dict[str, float]],
                                 toy1d_train_history: dict[str, np.ndarray]) -> dict:

    parallel_metrics = {}
    for name in toy1d_metrics["normal"]:
        parallel_metrics[name] = toy1d_train_history[f"{name}_val"][-1]

    relation = {}

    for name in toy1d_metrics["normal"]:
        relation[name] = toy1d_metrics["normal"][name] / parallel_metrics[name]

    total_metrics = {}
    total_metrics["autoregressive"] = toy1d_metrics["normal"]
    total_metrics["parallel"] = parallel_metrics
    total_metrics["proportion"] = relation

    return total_metrics


save_toy1d_autoregressive_metrics = function_variation({"metrics": source(
    "toy1d_autoregressive_metrics"), "metrics_name": value("autoregressive_metrics")}, "save_toy1d_autoregressive_metrics")(save_metrics)
