import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from world_machine_experiments.shared.acronyms import format_name
from world_machine_experiments.shared.save_metrics import get_values


def metrics_bar_with_errorbar(metrics: dict,
                              metric_name: str,
                              entry_caption_title: str,
                              palette: list | None = None,
                              errorbar_legend: str | None = None,
                              ax: Axes | None = None,
                              errorbar: bool = True) -> None:

    if ax is None:
        ax = plt.gca()

    n_entry = len(metrics)
    entry_names = list(metrics.keys())

    group_names = list(metrics[next(iter(metrics))]["means"].keys())

    df_metrics = []
    for i in range(n_entry):
        entry_name = entry_names[i]
        for j, group in enumerate(group_names):
            element = {}
            element["mean"] = get_values(
                metrics, ["means", group, metric_name])[i]
            element["std"] = get_values(
                metrics, ["stds", group, metric_name])[i]
            element["metric_name"] = format_name(group)
            element[entry_caption_title] = format_name(entry_name)

            df_metrics.append(element)
    df_metrics = pd.DataFrame(df_metrics)
    sns.barplot(df_metrics, x="metric_name", y="mean",
                hue=entry_caption_title, palette=palette, ax=ax)

    if errorbar:

        width = 1/(3+0.75)

        entry_names = df_metrics[entry_caption_title].unique()

        for i, name in enumerate(group_names):
            x_pos = i-(width*(len(entry_names)//2))
            for _, entry_name in enumerate(entry_names):
                entry = df_metrics[np.bitwise_and(
                    df_metrics["metric_name"] == format_name(name), df_metrics[entry_caption_title] == entry_name)]

                ax.errorbar(x_pos, entry["mean"], entry["std"], color="black",
                            ecolor="black", capsize=5, label=errorbar_legend)
                x_pos += width

                errorbar_legend = None

    plt.xlabel("Metric")
    plt.legend()


def metrics_boxplot(metrics: dict,
                    metric_name: str,
                    entry_caption_title: str,
                    palette: list | None = None,
                    ax: Axes | None = None) -> None:
    if ax is None:
        ax = plt.gca()

    df_metrics = []
    for name in metrics:
        for run_name in metrics[name]:

            for task in metrics[name][run_name]:
                element = {}
                element[entry_caption_title] = format_name(name)
                element["run_name"] = run_name
                element["task"] = format_name(task)
                for key in metrics[name][run_name][task]:
                    element[key] = metrics[name][run_name][task][key]

                df_metrics.append(element)

    df_metrics = pd.DataFrame(df_metrics)

    sns.boxplot(df_metrics, x="task", y=metric_name,
                hue=entry_caption_title, palette=palette, ax=ax)

    plt.xlabel("Metric")
    plt.legend()
