import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from world_machine_experiments.shared.acronyms import acronyms


def train_plots(train_history: dict[str, np.ndarray],
                log_y_axis: bool = True,
                x_axis: str | None = None,
                series_names: list[str] | None = None,
                plot_prefix: str | None = None) -> dict[str, Figure]:

    if "means" in train_history:
        with_std = True
        all_names = list(train_history["means"].keys())

        n_epoch = len(train_history["means"][all_names[0]])
    else:
        with_std = False
        all_names = list(train_history.keys())

        n_epoch = len(train_history[all_names[0]])

    names: set[str] = set()
    for name in all_names:
        names.add(name.removesuffix("_train").removesuffix("_val"))

    if "duration" in names:
        names.remove("duration")

    if x_axis is None:
        x_axis_values = range(1, n_epoch+1)
        x_label = "Epochs"
    else:
        if with_std:
            x_axis_values = train_history["means"][x_axis]
        else:
            x_axis_values = train_history[x_axis]
        x_label = x_axis.replace("_", " ").title()

        names.remove(x_axis)

    if series_names is None:
        series_names = ["train", "val"]
    elif len(series_names) == 0:
        series_names.append("")

    figures = {}
    for name in names:
        fig = plt.figure(dpi=300)

        negative = False

        for s_name in series_names:
            suffix = ""
            if s_name != "":
                suffix = "_"+s_name

            if with_std:
                plot_args = {"fmt": "o-", "capsize": 5.0, "markersize": 4}

                plt.errorbar(x_axis_values,
                             train_history["means"][name+suffix],
                             train_history["stds"][name+suffix],
                             label=(name+suffix).capitalize(), **plot_args)

                negative = negative or bool(
                    np.any(train_history["means"][name+suffix] <= 0))
            else:

                plt.plot(x_axis_values,
                         train_history[name+suffix], "o-", label=(name+suffix).capitalize())

                negative = negative or bool(
                    np.any(train_history[name+suffix] <= 0))

        name_format = name.replace("_", " ").title()

        for acro in acronyms:
            name_format = name_format.replace(acro.capitalize(), acro)

        plt.title(name_format)
        plt.xlabel(x_label)
        plt.ylabel("Metric")
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

        if log_y_axis:
            if not negative:
                plt.yscale("log")
            else:
                plt.yscale("asinh")

        plt.close()

        if plot_prefix:
            name = plot_prefix+"_"+name
        figures[name] = fig

    return figures


def multiple_train_plots(train_history, metrics):
    ...
