import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

acronyms = ["MSE"]


def parameter_variation_plots(train_history: dict[str, dict[str, np.ndarray]],
                              custom_plots: dict[str, list[str]] = None,
                              log_y_axis: bool = True) -> dict[str, Figure]:

    variation_names = list(train_history.keys())

    names: set[str] = set()
    for name in train_history[variation_names[0]]:
        names.add(name.removesuffix("_std").removesuffix(
            "_train").removesuffix("_val"))

    names.remove("duration")

    n_epoch = len(train_history[variation_names[0]]["duration"])
    epochs = range(1, n_epoch+1)

    colormap = plt.cm.nipy_spectral
    colors = colormap(np.linspace(0, 1, len(train_history)))
    color_map = {variation_names[i]: colors[i]
                 for i in range(len(variation_names))}

    plot_args = {"fmt": "o-", "capsize": 5.0, "markersize": 4}

    plot_combinations = {"": variation_names}
    if custom_plots is not None:
        plot_combinations.update(custom_plots)

    figures = {}
    for combination_name in plot_combinations:
        combination = plot_combinations[combination_name]

        for split in ["train", "val"]:
            for name in names:
                key = name+"_"+split

                fig, _ = plt.subplots(dpi=300)

                for variation_name in combination:

                    plt.errorbar(epochs,
                                 train_history[variation_name][key],
                                 train_history[variation_name][key+"_std"],
                                 label=variation_name,
                                 color=color_map[variation_name],
                                 **plot_args)

                name_format = name.replace("_", " ").title()

                for acro in acronyms:
                    name_format = name_format.replace(acro.capitalize(), acro)

                plt.suptitle(name_format)
                plt.title(split)
                plt.xlabel("Epochs")
                plt.ylabel("Metric")
                plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

                if log_y_axis:
                    plt.yscale("log")

                plt.close()

                if combination_name != "":
                    plot_name = combination_name+"_"+key
                else:
                    plot_name = key

                figures[plot_name] = fig

    return figures
