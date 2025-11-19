
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hamilton.function_modifiers import extract_fields, source, value
from matplotlib.figure import Figure
from scipy.stats import wilcoxon

from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.acronyms import format_name
from world_machine_experiments.shared.save_metrics import save_metrics
from world_machine_experiments.shared.save_plots import save_plots

from ._shared import *
from ._shared import _format_variable_name, _get_disjoint_vars


def _impact_test(variations_df: pd.DataFrame,
                 disjoint_groups: list[set[str]],
                 variable: str,
                 task: str) -> dict:

    x: pd.DataFrame = variations_df[variations_df[variable] == False]
    y: pd.DataFrame = variations_df[variations_df[variable] == True]

    disjoint_vars = _get_disjoint_vars(variable, disjoint_groups)

    for var in disjoint_vars:
        x = x[x[var] == False]

    for var in variable_names:
        if var == variable or var in disjoint_vars:
            continue

        x = x.sort_values(by=var, kind="stable")
        y = y.sort_values(by=var, kind="stable")

    x_mask = x["mask"].to_numpy()
    y_mask = y["mask"].to_numpy()
    mask = np.bitwise_and(x_mask, y_mask)

    x: np.ndarray = x[task].to_numpy()
    y: np.ndarray = y[task].to_numpy()

    x = x[mask]
    y = y[mask]

    result = wilcoxon(x, y, nan_policy="omit")

    diff = (y - x)
    diff = diff[np.bitwise_not(np.isnan(diff))]
    impact = np.median(diff)

    return {"pvalue": result.pvalue, "impact": impact, "diff": diff}


@extract_fields({"impact_test_df": pd.DataFrame, "impact_test_full_df": pd.DataFrame})
def impact_test_dfs(variations_df: pd.DataFrame,
                    disjoint_groups: list[set[str]]) -> dict[str, pd.DataFrame]:
    test_data = []
    test_full_data = []

    for var in variable_names:

        for task in task_names+["duration"]:

            if task != "duration":
                result = _impact_test(variations_df,
                                      disjoint_groups, var, f"{task}_mse")
            else:
                result = _impact_test(variations_df, disjoint_groups,
                                      var, "duration")

            item = {"variable": var, "task": format_name(task)}
            item["p_value"] = result["pvalue"]
            item["impact"] = result["impact"]
            item["failed"] = result["pvalue"] >= 0.05

            test_data.append(item)

            for d in result["diff"]:
                item = item.copy()
                item["diff"] = d

                test_full_data.append(item)

    df_test = pd.DataFrame(test_data)
    df_test_full = pd.DataFrame(test_full_data)

    return {"impact_test_df": df_test, "impact_test_full_df": df_test_full}


save_impact_test_df = function_variation({
    "metrics": source("impact_test_df"),
    "metrics_name": value("impact_test_df")},
    "save_impact_test_df")(save_metrics)

save_impact_test_full_df = function_variation({
    "metrics": source("impact_test_full_df"),
    "metrics_name": value("impact_test_full_df")},
    "save_impact_test_full_df")(save_metrics)


def _impact_plot(df_test_full, df_failed,
                 variables_to_exclude: None | list[str],
                 tasks_to_exclude: None | list[str],
                 palette=None, hue="task",
                 vertical_separator: bool = True):

    if tasks_to_exclude is not None:
        for task in tasks_to_exclude:
            df_test_full = df_test_full[df_test_full["task"] != task]
            df_failed = df_failed[df_failed["task"] != task]

    if variables_to_exclude is not None:
        for variable in variables_to_exclude:
            df_test_full = df_test_full[df_test_full["variable"] != variable]
            df_failed = df_failed[df_failed["variable"] != variable]

    task_names_formatted = list(df_test_full["task"].unique())
    variables = list(df_test_full["variable"].unique())

    bar_width = 0.75/len(task_names_formatted)

    ax = plt.gca()

    sns.boxplot(df_test_full, x="variable", y="diff", hue=hue,
                ax=ax, width=0.75, palette=palette,
                flierprops={"markersize": 1},  boxprops={"edgecolor": 'none'})

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    plt.hlines(0, xlim[0], xlim[1], color="black", lw=1, zorder=-1)

    for i in range(len(df_failed)):
        variable = df_failed.iloc[i]["variable"]
        task = df_failed.iloc[i]["task"]

        index = variables.index(variable)

        task_offset = task_names_formatted.index(task)
        task_offset -= len(task_names_formatted)//2
        task_offset *= bar_width

        index += task_offset

        label = None
        if i == 0:
            label = "No Statistical\nRelevance"  # (p ≥ 0.05)"
        plt.bar([index, index],  ylim, alpha=0.5,
                color="gray", width=bar_width, label=label, zorder=-2)

    first = True
    for variable in variables:
        if variable in ["AC_1"]:
            label = None
            if first:
                label = "High Divergence\nProbability"
                first = False

            index = variables.index(variable)

            plt.bar([index, index], ylim, alpha=0.2, color="black", width=0.75,
                    label=label, zorder=-1, hatch="xxxx", edgecolor="black", linewidth=0, fill=False)

    if vertical_separator:
        for i in range(len(variables)-1):
            plt.vlines(i+0.5, -ylim[0], -ylim[1], color="black")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.xticks(plt.xticks()[0], _format_variable_name(variables))


def task_impact_plots(impact_test_df: pd.DataFrame,
                      impact_test_full_df: pd.DataFrame) -> dict[str, Figure]:

    mid_index = len(variable_names)//2
    failed = impact_test_df[impact_test_df["failed"]][["task", "variable"]]

    fig, axs = plt.subplots(2, dpi=600, figsize=(6.4, 1.25*4.8))

    plt.sca(axs[0])
    _impact_plot(impact_test_full_df, failed,
                 variable_names[mid_index:], ["Duration"])

    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)

    handles, labels = axs[0].get_legend_handles_labels()
    for i in range(len(labels)):
        labels[i] = labels[i].replace(" ", "\n")
    axs[0].legend(handles, labels, bbox_to_anchor=(1.01, 1), borderaxespad=0)

    plt.title("Variable Impact on Tasks")

    plt.sca(axs[1])
    _impact_plot(impact_test_full_df, failed,
                 variable_names[:mid_index], ["Duration"])

    plt.xlabel("Variable")

    axs[0].set_xlabel("")
    axs[1].get_legend().remove()

    y_max = 0

    for ax in axs:
        plt.sca(ax)

        ax.set_ylabel("")
        ax.set_yscale("asinh", linear_width=0.0025)
        plt.grid(which="both", axis="y", ls="-", color="black", alpha=0.25)

        yticks = plt.yticks()
        plt.yticks(list(yticks[0][:3])+list(yticks[0]
                   [4:]), yticks[1][:3]+yticks[1][4:])

        y_max = np.max(np.abs(ax.get_ylim()))

    for ax in axs:
        ax.set_ylim(-y_max, y_max)

    big_ax = fig.add_subplot(111, frameon=False)
    big_ax.tick_params(labelcolor='none', which='both',
                       top=False, bottom=False, left=False, right=False)
    big_ax.set_ylabel("MSE Impact (Negative is Better)", labelpad=20)

    return {"variable_impact_on_tasks": fig}


save_task_impact_plots = function_variation({
    "plots": source("task_impact_plots")},
    "save_task_impact_plots")(save_plots)


def duration_impact_plots(impact_test_df: pd.DataFrame,
                          impact_test_full_df: pd.DataFrame) -> dict[str, Figure]:

    failed = impact_test_df[impact_test_df["failed"]][["task", "variable"]]

    fig = plt.figure(dpi=600)

    plot_palette = [palette[2] if negative_impact else palette[1] for negative_impact in (
        impact_test_df[impact_test_df["task"] == "Duration"]["impact"] < 0).to_numpy()]
    _impact_plot(impact_test_full_df, failed, None,
                 format_name(task_names), plot_palette, "variable", False)

    ylim = plt.ylim()

    for i in range(n_variable-1):
        plt.vlines(i+0.5, ylim[0], ylim[1], "black", linewidth=0.5)

    plt.ylim(ylim)
    ax = plt.gca()
    ax.tick_params(axis='x', length=0)

    handles, labels = ax.get_legend_handles_labels()
    for i in range(len(labels)):
        labels[i] = labels[i].replace(" ", "\n")
    ax.legend(handles, labels, bbox_to_anchor=(1.01, 1), borderaxespad=0)

    plt.title("Variable Impact on Duration")
    plt.ylabel("Time Impact [s] (Negative is Better)")
    plt.xlabel("Variable")

    return {"variable_impact_on_duration": fig}


save_duration_impact_plots = function_variation({
    "plots": source("duration_impact_plots")},
    "save_duration_impact_plots")(save_plots)
