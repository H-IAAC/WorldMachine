import os
from typing import Any

from pydantic import BaseModel, ImportString, create_model


def _make_model(v, name):
    if type(v) is dict:
        return create_model(name, **{k: _make_model(v, k) for k, v in v.items()}), ...
    elif type(v) is type:
        return ImportString, v
    return type(v), v


def make_model(v: dict, name: str):
    return _make_model(v, name)[0]


def save_parameters(n_run: int,
                    base_seed: int,
                    output_dir: str,
                    parameters: dict[str, Any]) -> dict:

    log = {"n_run": n_run,
           "base_seed": base_seed,
           "parameters": parameters}

    model: BaseModel = make_model(log, "ParametersModel").model_validate(log)
    model_json = model.model_dump_json(indent=4)

    file_path = os.path.join(output_dir, "parameters.json")

    with open(file_path, "w") as file:
        file.write(model_json)

    return {"path": file_path}
