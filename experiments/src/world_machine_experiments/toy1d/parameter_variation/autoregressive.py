import json
import os


def variation_autoregressive_metrics(experiment_paths: dict[str, str]) -> dict[str, dict[str, dict[str, float]]]:
    metrics = {"autoregressive": {}, "parallel": {}, "proportion": {}}

    for metric_name in metrics:
        metrics[metric_name] = {"train": {}, "val": {}}

    for experiment_name in experiment_paths:
        path = experiment_paths[experiment_name]
        path = os.path.join(path, "run_0", "autoregressive_metrics.json")

        with open(path, "r") as file:
            metrics_experiment = json.load(file)

        for metric_name in metrics:
            for split in ["train", "val"]:
                metrics[metric_name][split][experiment_name] = metrics_experiment[metric_name][split]

    return metrics


def save_variation_autoregressive_metrics(variation_autoregressive_metrics: dict[str, dict[str, dict[str, float]]], output_dir: str) -> dict:
    path = os.path.join(output_dir, "autoregressive_metrics.json")

    with open(path, "w") as file:
        json.dump(variation_autoregressive_metrics, file)

    return {"path": path}
