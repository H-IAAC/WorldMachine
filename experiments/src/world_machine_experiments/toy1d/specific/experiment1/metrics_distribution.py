
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hamilton.function_modifiers import source, value
from matplotlib.figure import Figure
from scipy.stats import spearmanr

from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.acronyms import format_name
from world_machine_experiments.shared.save_metrics import save_metrics
from world_machine_experiments.shared.save_plots import save_plots

from ._shared import *


def task_distribution_plots(variations_df: pd.DataFrame) -> dict[str, Figure]:
    figures = {}

    for metric in metric_names:
        items = []

        for i in range(len(variations_df)):
            if variations_df["mask"].iloc[i]:
                for task in task_names:
                    item = {}
                    item["task"] = format_name(task).replace(" ", "\n")
                    item["value"] = variations_df[f"{task}_{metric}"].iloc[i]

                    items.append(item)

        df_metrics = pd.DataFrame(items)

        fig = plt.figure(dpi=600)
        sns.violinplot(df_metrics, x="task", y="value", hue="task")

        plt.yscale("log")
        plt.xlabel("Task")
        plt.ylabel(format_name(metric))
        plt.title("Metric Distribution Across Tasks")

        figures[f"metric_distribution_{metric}"] = fig

    return figures


save_task_distribution_plots = function_variation({"plots": source(
    "task_distribution_plots")}, "save_task_distribution_plots")(save_plots)


def tasks_correlation(variations_df: pd.DataFrame) -> dict[str, dict]:
    correlations = {}

    for metric in metric_names:
        correlation_variables = []
        for task in task_names:
            correlation_variables.append(
                variations_df[variations_df["mask"]][f"{task}_{metric}"].to_numpy())
        correlation_variables = np.array(correlation_variables)

        pearson_correlation = np.corrcoef(correlation_variables)
        spearman_correlation = spearmanr(
            correlation_variables, axis=1).statistic

        correlations[metric] = {
            "pearson": pearson_correlation,
            "spearman": spearman_correlation
        }

    return correlations


save_tasks_correlation = function_variation({
    "metrics": source("tasks_correlation"),
    "metrics_name": value("tasks_correlation")},
    "save_tasks_correlation")(save_metrics)


def tasks_correlation_plots(tasks_correlation: dict[str, dict]) -> dict[str, Figure]:
    n_task = len(task_names)

    figures = {}

    for metric in metric_names:
        for correlation_name in ["pearson", "spearman"]:
            correlation = tasks_correlation[metric][correlation_name]

            fig = plt.figure(dpi=600)

            cmap = mcolors.ListedColormap(["0", "0.2", "0.4", "0.6", "0.8"])

            plt.imshow(correlation, cmap=cmap, vmin=0, vmax=1)
            plt.colorbar()

            xlim = plt.xlim()
            ylim = plt.ylim()

            for i in range(4):
                plt.hlines(i+0.5, -0.5, 4.5, "black")
                plt.vlines(i+0.5, -0.5, 4.5, "black")

            ax = plt.gca()

            for i in range(len(correlation)):
                for j in range(i, len(correlation)):
                    ax.text(j, i, np.around(correlation[i, j], decimals=3),
                            ha="center", va="center", color="black")

            xtick_labels = format_name(task_names)
            for i in range(len(xtick_labels)):
                xtick_labels[i] = xtick_labels[i].replace(" ", "\n")

            plt.xticks(range(5), xtick_labels)
            plt.yticks(range(5), xtick_labels)

            inf_mask = np.tri(n_task, n_task, -1)
            plt.imshow(inf_mask, alpha=inf_mask, cmap="gray", vmin=0, vmax=1)

            plt.xlim(xlim)
            plt.ylim(ylim)

            plt.title(
                f"{format_name(correlation_name)} Correlation Between\nTasks' {format_name(metric)}")

            figures[f"{correlation_name}_{metric}_between_tasks"] = fig

    return figures


save_tasks_correlation_plots = function_variation({"plots": source(
    "tasks_correlation_plots")}, "save_tasks_correlation_plots")(save_plots)
