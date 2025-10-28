import glob
import json
import os


def load_multiple_metrics(output_dir: str, metrics_name: str) -> list[dict]:
    paths = glob.glob(os.path.join(output_dir, "*"))

    result: list[dict] = []
    for path in paths:
        if os.path.isdir(path):
            file_path = os.path.join(path, f"{metrics_name}.json")

            with open(file_path, "r") as file:
                try:
                    result.append(json.load(file))
                except Exception as e:
                    print(f"ERROR reading metrics file {file_path}")
                    raise e

    return result
