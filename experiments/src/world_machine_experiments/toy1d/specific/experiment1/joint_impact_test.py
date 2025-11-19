
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hamilton.function_modifiers import source, value
from matplotlib.figure import Figure
from scipy.stats import wilcoxon

from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.acronyms import format_name
from world_machine_experiments.shared.save_metrics import save_metrics
from world_machine_experiments.shared.save_plots import save_plots

from ._shared import *


def _joint_impact_test(variations_df: pd.DataFrame,
                       disjoint_groups: list[set[str]],
                       var1: str,
                       var2: str,
                       task: str) -> dict:
    df = variations_df

    v1v2 = df[np.bitwise_and(df[var1], variations_df[var2])]
    v1nv2 = df[np.bitwise_and(df[var1], np.bitwise_not(df[var2]))]
    nv1v2 = df[np.bitwise_and(np.bitwise_not(df[var1]),  df[var2])]
    nv1nv2 = df[np.bitwise_and(
                np.bitwise_not(df[var1]),
                np.bitwise_not(df[var2])
                )]

    disjoint_vars = set()
    for group in disjoint_groups:
        if var1 in group or var2 in group:
            disjoint_vars.update(group)

    for var in [var1, var2]:
        if var in disjoint_vars:
            disjoint_vars.remove(var)

    sets = [v1v2, v1nv2, nv1v2, nv1nv2]

    for var in disjoint_vars:
        for i in range(len(sets)):
            sets[i] = sets[i][sets[i][var] == False]

    for var in variable_names:
        if var in [var1, var2] or var in disjoint_vars:
            continue

        for i in range(len(sets)):
            sets[i] = sets[i].sort_values(by=var, kind="stable")

    mask = np.ones(len(sets[0]), bool)
    for i in range(4):
        mask = np.bitwise_and(mask, sets[i]["mask"].to_numpy())

    for i in range(4):
        sets[i] = sets[i][task].to_numpy()
        sets[i] = sets[i][mask]

    v1v2 = sets[0]
    v1nv2 = sets[1]
    nv1v2 = sets[2]
    nv1nv2 = sets[3]

    diff = v1v2 - v1nv2 - nv1v2 + nv1nv2
    diff = diff[np.bitwise_not(np.isnan(diff))]

    result = wilcoxon(diff, nan_policy="omit")

    impact = np.median(diff)

    return {"pvalue": result.pvalue, "impact": impact, "diff": diff}


def joint_impact(variations_df: pd.DataFrame,
                 disjoint_groups: list[set[str]],
                 impact_test_df: pd.DataFrame) -> dict:

    result = {}
    for task in task_names:
        result[task] = {}

        joint_impact = np.zeros((n_variable, n_variable))

        joint_impact_pvalue = np.zeros((n_variable, n_variable))

        for i in range(n_variable):
            for j in range(i+1, n_variable):
                var1 = variable_names[i]
                var2 = variable_names[j]

                disjoint = False
                for group in disjoint_groups:
                    if var1 in group and var2 in group:
                        disjoint = True
                        break

                if disjoint:
                    continue

                test_result = _joint_impact_test(
                    variations_df, disjoint_groups, var1, var2, f"{task}_mse")

                joint_impact[i, j] = test_result["impact"]
                joint_impact[j, i] = test_result["impact"]

                joint_impact_pvalue[i, j] = test_result["pvalue"]
                joint_impact_pvalue[j, i] = test_result["pvalue"]

        for i, var in enumerate(variable_names):
            joint_impact[i, i] = impact_test_df[np.bitwise_and(
                impact_test_df["variable"] == var, impact_test_df["task"] == format_name(task))]["impact"].iloc[0]
            joint_impact_pvalue[i, i] = impact_test_df[np.bitwise_and(
                impact_test_df["variable"] == var, impact_test_df["task"] == format_name(task))]["p_value"].iloc[0]

        result[task]["impact"] = joint_impact
        result[task]["p_value"] = joint_impact_pvalue

    return result


save_joint_impact = function_variation({
    "metrics": source("joint_impact"),
    "metrics_name": value("joint_impact")},
    "save_joint_impact")(save_metrics)


def disjoint_mask(disjoint_groups: list[set[str]]) -> np.ndarray:
    result = np.zeros((n_variable, n_variable))

    for dg in disjoint_groups:
        dg = list(dg)
        for i in range(len(dg)):
            for j in range(i+1, len(dg)):
                var_i = np.argwhere(variable_names == dg[i]).item()
                var_j = np.argwhere(variable_names == dg[j]).item()

                result[var_i, var_j] = 1
                result[var_j, var_i] = 1

    return result


def joint_impact_plots(joint_impact: dict,
                       disjoint_mask: np.ndarray) -> dict[str, Figure]:

    figures = {}
    for task in task_names:
        if task in ["prediction_shallow", "prediction"]:
            scale = 1e3
            scale_exponent = 3
        elif task in ["normal"]:
            scale = 1e5
            scale_exponent = 5
        else:
            scale = 1e4
            scale_exponent = 4

        fig = plt.figure(dpi=600)

        impact = joint_impact[task]["impact"]
        p_value = joint_impact[task]["p_value"]

        range_limit = 2*np.std(impact)

        ji_img = plt.imshow(scale*impact,  cmap="bwr", vmin=-
                            scale*range_limit, vmax=scale*range_limit)
        plt.colorbar(ji_img, pad=0.08, fraction=0.042)

        plt.imshow(0.5*np.ones_like(impact), alpha=(p_value >=
                   0.05).astype(np.float32), vmin=0, vmax=1, cmap="gray")
        plt.imshow(np.zeros_like(impact), alpha=disjoint_mask,
                   vmin=0, vmax=1, cmap="gray")

        ax = plt.gca()
        ax.set_xticks(np.arange(0, n_variable, 1),
                      variable_names_f, rotation=45)
        ax.set_yticks(np.arange(0, n_variable, 1),
                      variable_names_f, rotation=0)

        ax2 = ax.secondary_yaxis("right")
        ax2.set_yticks(np.arange(0, n_variable, 1),
                       variable_names_f, rotation=0)

        ax2 = ax.secondary_xaxis("top")
        ax2.set_xticks(np.arange(0, n_variable, 1),
                       variable_names_f, rotation=45)

        inf_mask = np.tri(n_variable, n_variable, -1)
        plt.imshow(inf_mask, alpha=inf_mask, cmap="gray", vmin=0, vmax=1)

        divisor_color = "white"

        for pos in [1.5, 3.5, 4.5, 5.5, 7.5, 11.5, 15.5]:
            for func in [plt.hlines, plt.vlines]:
                func(pos, -0.5, n_variable-0.5, color=divisor_color)

        for i in range(n_variable):
            for j in range(i, n_variable):
                if p_value[i, j] < 0.05 and not disjoint_mask[i, j]:
                    text = str(int(np.around(scale*impact[i, j])))

                    fontsize = 10
                    if len(text) > 2:
                        fontsize = 8

                    color = "black"
                    if abs(scale*impact[i, j]) > (3/4)*scale*range_limit:
                        color = "white"

                    ax.text(j, i,
                            text,
                            ha="center", va="center", color=color, fontsize=fontsize)

        plt.xlabel("Variable")
        plt.ylabel("Variable")
        plt.title("Variables Individual and Synergetic Impact\n" +
                  format_name(task)+" MSE " + r"$Impact\times1E-$"+f"{scale_exponent}")

        pvalue_patch = mpatches.Patch(
            color='gray', label="No Statistical Relevance (p ≥ 0.05)")
        disjoint_patch = mpatches.Patch(
            color='black', label="Disjoint Variables")
        plt.legend(handles=[pvalue_patch, disjoint_patch],
                   bbox_to_anchor=(0.7, -0.2), borderaxespad=0)

        figures[f"joint_impact_{task}"] = fig

    return figures


save_joint_impact_plots = function_variation({
    "plots": source("joint_impact_plots")},
    "save_joint_impact_plots")(save_plots)


def filtered_marginal_impact(joint_impact: dict,
                             disjoint_mask: np.ndarray) -> dict:

    variables_mask = variable_names != "AC_1"

    result = {}
    for task in task_names:
        joint_impact_filtered = joint_impact[task]["impact"].copy()
        joint_impact_filtered[joint_impact[task]["p_value"] >= 0.05] = 0
        joint_impact_filtered[disjoint_mask.astype(bool)] = 0

        joint_impact_filtered = joint_impact_filtered[variables_mask][:, variables_mask]
        marginal_impact = joint_impact_filtered.sum(axis=0)

        result[task] = dict(
            zip(variable_names[variables_mask], marginal_impact))

    return result


save_filtered_marginal_impact = function_variation({
    "metrics": source("filtered_marginal_impact"),
    "metrics_name": value("filtered_marginal_impact")},
    "save_filtered_marginal_impact")(save_metrics)


def filtered_marginal_impact_plots(filtered_marginal_impact: dict,
                                   joint_impact: dict) -> dict[str, Figure]:
    filtered_n_variable = n_variable-1
    variables_mask = variable_names != "AC_1"

    figures = {}
    for task in task_names:
        individual_impact = joint_impact[task]["impact"].copy()
        p_value = joint_impact[task]["p_value"]

        individual_impact[p_value >= 0.05] = 0
        individual_impact = individual_impact.diagonal()
        individual_impact = individual_impact[variables_mask]

        fig = plt.figure(dpi=600)
        plt.bar(np.arange(filtered_n_variable)-0.15,
                individual_impact, label="Individual Impact", width=0.3)
        plt.bar(np.arange(filtered_n_variable)+0.15,
                filtered_marginal_impact[task].values(), label="Marginal Impact",  width=0.3)

        plt.yscale("asinh", linear_width=0.0025)

        xlim = plt.xlim()
        ylim = plt.ylim()

        plt.hlines(0, xlim[0], xlim[1], "black", linewidth=1)

        for i in range(filtered_n_variable-1):
            plt.vlines(i+0.5, ylim[0], ylim[1], "black", linewidth=0.5)

        plt.xlim((xlim[0]+0.5, xlim[1]-0.5))
        plt.ylim(ylim)

        plt.xticks(np.arange(filtered_n_variable),
                   variable_names_f[variables_mask])

        ax = plt.gca()
        ax.tick_params(axis='x', length=0)

        plt.legend()
        plt.xlabel("Variable")
        plt.ylabel("MSE Impact (Negative is Better)")
        plt.title(
            f"Variable Individual x Marginal Impact\nTask: {format_name(task)}")

        figures[f"individual_x_marginal_impact_{task}"] = fig

    return figures


save_filtered_marginal_impact_plots = function_variation({
    "plots": source("filtered_marginal_impact_plots")},
    "save_filtered_marginal_impact_plots")(save_plots)
