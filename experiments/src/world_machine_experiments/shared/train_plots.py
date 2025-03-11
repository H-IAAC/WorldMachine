import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from world_machine_experiments.shared.acronyms import acronyms


def train_plots(train_history: dict[str, np.ndarray],
                log_y_axis: bool = True) -> dict[str, Figure]:
    names: set[str] = set()

    with_std = False

    for name in train_history:
        if name.endswith("_std"):
            with_std = True

        names.add(name.removesuffix("_std").removesuffix(
            "_train").removesuffix("_val"))

    names.remove("duration")

    n_epoch = len(train_history["duration"])
    epochs = range(1, n_epoch+1)

    figures = {}
    for name in names:
        fig = plt.figure(dpi=300)

        train_hist = train_history[name+"_train"]
        val_hist = train_history[name+"_val"]

        if with_std:
            train_hist_std = train_history[name+"_train_std"]
            val_hist_std = train_history[name+"_val_std"]

            plot_args = {"fmt": "o-", "capsize": 5.0, "markersize": 4}

            plt.errorbar(epochs, train_hist, train_hist_std,
                         label="Train", **plot_args)
            plt.errorbar(epochs, val_hist, val_hist_std,
                         label="Validation", **plot_args)
        else:
            plt.plot(epochs, train_hist, "o-", label="Train")
            plt.plot(epochs, val_hist, "o-", label="Validation")

        name_format = name.replace("_", " ").title()

        for acro in acronyms:
            name_format = name_format.replace(acro.capitalize(), acro)

        plt.title(name_format)
        plt.xlabel("Epochs")
        plt.ylabel("Metric")
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

        if log_y_axis:
            plt.yscale("log")

        plt.close()

        figures[name] = fig

    return figures
