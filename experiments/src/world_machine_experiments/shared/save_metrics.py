import json
import os

from hamilton.function_modifiers import datasaver


@datasaver()
def save_metrics(metrics: dict, output_dir: str, metrics_name: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, metrics_name+".json")

    with open(file_path, "w") as file:
        json.dump(metrics, file)

    return {"path": file_path}
