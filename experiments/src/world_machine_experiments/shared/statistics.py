import numpy as np


def _flatten(d, parent_key=()):
    items = {}
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, dict):
            items.update(_flatten(v, new_key))
        else:
            items[new_key] = v
    return items


def _unflatten(keys, values):
    result = {}
    for key_tuple, value in zip(keys, values):
        d = result
        for k in key_tuple[:-1]:
            d = d.setdefault(k, {})
        d[key_tuple[-1]] = value
    return result


def consolidated_metrics(metrics: list[dict]) -> dict:
    flat_data = [_flatten(d) for d in metrics]

    keys = list(flat_data[0])

    values = [[d[key] for key in keys] for d in flat_data]
    values = np.array(values)

    means = values.mean(axis=0)
    stds = values.std(axis=0)

    means = _unflatten(keys, means)
    stds = _unflatten(keys, stds)

    result = {"means": means, "stds": stds}

    return result
