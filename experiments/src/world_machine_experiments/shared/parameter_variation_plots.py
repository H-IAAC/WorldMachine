import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from world_machine_experiments.shared.acronyms import acronyms


def parameter_variation_plots(train_history: dict,
                              custom_plots: dict[str, list[str]] = None,
                              log_y_axis: bool = True,
                              x_axis: str | None = None,
                              series_names: list[str] | None = None,
                              plot_prefix: str | None = None) -> dict[str, Figure]:

    variation_names = list(train_history.keys())

    all_names = list(train_history[variation_names[0]]["means"].keys())

    names: set[str] = set()
    for name in all_names:
        names.add(name.removesuffix("_train").removesuffix("_val"))

    for name in ["duration", "mask_sensory_percentage"]:
        if name in names:
            names.remove(name)

    if x_axis is None:
        n_epoch = len(train_history[variation_names[0]]["means"]["duration"])
        x_axis_values = range(1, n_epoch+1)
        x_label = "Epochs"
    else:

        x_axis_values = train_history[variation_names[0]]["means"][x_axis]
        x_label = x_axis.replace("_", " ").title()

    colormap = plt.cm.nipy_spectral
    colors = colormap(np.linspace(0, 1, len(train_history)))
    color_map = {variation_names[i]: colors[i]
                 for i in range(len(variation_names))}

    plot_args = {"fmt": "o-", "capsize": 5.0, "markersize": 4}

    plot_combinations = {"": variation_names}
    if custom_plots is not None:
        plot_combinations.update(custom_plots)

    if series_names is None:
        series_names = ["train", "val"]
    elif len(series_names) == 0:
        series_names.append("")

    figures = {}
    for combination_name in plot_combinations:
        combination = plot_combinations[combination_name]

        for s_name in series_names:
            suffix = ""
            if s_name != "":
                suffix = "_"+s_name

            for name in names:
                key = name+suffix

                fig, _ = plt.subplots(dpi=300)
                negative = False

                for variation_name in combination:

                    if variation_name not in train_history or key not in train_history[variation_name]["means"]:
                        continue

                    plt.errorbar(x_axis_values,
                                 train_history[variation_name]["means"][key],
                                 train_history[variation_name]["stds"][key],
                                 label=variation_name,
                                 color=color_map[variation_name],
                                 **plot_args)

                    negative = negative or bool(
                        np.any(train_history[variation_name]["means"][key] <= 0))

                name_format = name.replace("_", " ").title()

                for acro in acronyms:
                    name_format = name_format.replace(acro.capitalize(), acro)

                if s_name != "":
                    plt.suptitle(name_format)
                    plt.title(s_name)
                else:
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

                if combination_name != "":
                    plot_name = combination_name+"_"+key
                else:
                    plot_name = key

                if plot_prefix:
                    plot_name = plot_prefix+"_"+plot_name
                figures[plot_name] = fig

    return figures
