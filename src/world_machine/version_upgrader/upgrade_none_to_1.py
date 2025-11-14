from typing import Any

# Upgrade from model version = None to model version = 1


def update_key(x: dict[str:Any], original_key: str, new_key: str) -> None:
    x[new_key] = x[original_key]
    del x[original_key]


def upgrade_none_to_1(state: dict):
    dicts: list[dict[str, Any]] = [state]

    if "_modules" in state:
        dicts.append(state["_modules"])

    for x in dicts:
        keys = list(x.keys())
        for name in keys:
            updated_name = name.replace("sensorial", "sensory")
            updated_name = updated_name.replace("dimension", "channel")

            if updated_name != name:
                update_key(x, name, updated_name)

    state["_compatibility_version"] = 1
