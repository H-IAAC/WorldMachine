import json
import os
import re

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from hamilton.function_modifiers import (
    datasaver, extract_fields, source, value)
from matplotlib.figure import Figure
from scipy.stats import pearsonr, spearmanr, wilcoxon

from world_machine.train.stages import StateSaveMethod
from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.acronyms import format_name
from world_machine_experiments.shared.save_metrics import (
    get_values, load_multiple_metrics, save_metrics)
from world_machine_experiments.shared.save_plots import save_plots
from world_machine_experiments.toy1d.dimensions import Dimensions

plt.rcParams["mathtext.default"] = "regular"

# Default Data

palette = ["#60BF60", "#D36767", "#6060BF"]

task_names = ["normal", "use_state", "prediction",
              "prediction_shallow", "prediction_local"]

metric_names = ["mse", "0.1sdtw"]

variable_names = np.array(["SB_1", "SB_2", "SM_1", "SM_2", "AC_1", "MD_1", "NA_1", "NA_2",
                           "RP_1", "RP_2", "RP_3", "RP_4",
                           "RF_1", "RF_2", "RF_3", "RF_4",
                           "LM_1"])


n_variable = len(variable_names)


def _format_variable_name(var: str | list):
    if not isinstance(var, str):
        return np.array([_format_variable_name(v) for v in var])

    return f"${var}$"


variable_names_f = _format_variable_name(variable_names)


# Load variations data

metrics = function_variation({"output_dir": source("data_dir"),
                              "metrics_name": value("toy1d_metrics")},
                             "metrics")(load_multiple_metrics)

parameters = function_variation({"output_dir": source("data_dir"),
                                 "metrics_name": value("parameters")},
                                "parameters")(load_multiple_metrics)

train_history = function_variation({"output_dir": source("data_dir"),
                                    "metrics_name": value("train_history")},
                                   "train_history")(load_multiple_metrics)


def _add_diverge_mask(df: pd.DataFrame) -> None:
    mask = np.ones(len(df), bool)
    for metric in task_names:

        mask = np.bitwise_and(mask, np.bitwise_not(df[metric+"_mse"].isna()))

        data = df[f"{metric}_mse"].to_numpy()

        data_mask = np.bitwise_not(np.isnan(data))
        data_mask = np.bitwise_and(data_mask, data < 1)

        data_max = data[data_mask].mean() + 3*data[data_mask].std()

        mask = np.bitwise_and(mask, np.bitwise_not(np.isnan(data)))
        mask = np.bitwise_and(mask, data < data_max)

    df["mask"] = mask


def _add_variables(df: pd.DataFrame) -> None:
    df["SB_1"] = np.bitwise_and(df["n_segment"] == 2, df["fast_forward"] == False)  # nopep8
    df["SB_2"] = np.bitwise_and(df["n_segment"] == 2, df["fast_forward"] == True)  # nopep8

    df["SM_1"] = df["state_save_method"] == StateSaveMethod.MEAN.value  # nopep8
    df["SM_2"] = df["check_input_masks"] == True  # nopep8

    df["AC_1"] = pd.isnull(df["state_activation"])  # nopep8

    df["MD_1"] = df["block_configuration"].map(lambda x: np.all(x == [Dimensions.MEASUREMENT.value, Dimensions.STATE_INPUT.value]))  # nopep8

    df["NA_1"] = df["noise_config"].map(lambda x: x is not None and "state" in x)  # nopep8
    df["NA_2"] = df["noise_config"].map(lambda x: x is not None and "measurement" in x)  # nopep8

    df["RP_1"] = np.bitwise_and(df["recall_stride_past"] == 1, df["recall_n_past"] == 1)  # nopep8
    df["RP_2"] = np.bitwise_and(df["recall_stride_past"] == 1, df["recall_n_past"] == 5)  # nopep8
    df["RP_3"] = np.bitwise_and(df["recall_stride_past"] == 3, df["recall_n_past"] == 1)  # nopep8
    df["RP_4"] = np.bitwise_and(df["recall_stride_past"] == 3, df["recall_n_past"] == 5)  # nopep8

    df["RF_1"] = np.bitwise_and(df["recall_stride_future"] == 1, df["recall_n_future"] == 1)  # nopep8
    df["RF_2"] = np.bitwise_and(df["recall_stride_future"] == 1, df["recall_n_future"] == 5)  # nopep8
    df["RF_3"] = np.bitwise_and(df["recall_stride_future"] == 3, df["recall_n_future"] == 1)  # nopep8
    df["RF_4"] = np.bitwise_and(df["recall_stride_future"] == 3, df["recall_n_future"] == 5)  # nopep8

    df["LM_1"] = np.bitwise_not(pd.isnull(df["local_chance"]))  # nopep8


def variations_df(metrics: dict[str, dict],
                  parameters: dict[str, dict],
                  train_history: dict[str, dict]) -> pd.DataFrame:
    variations_data = []

    for name in parameters:
        item = parameters[name]["parameters"]
        item["name"] = name

        for m in task_names:
            m_1 = m
            m_2 = m

            for criterion in ["mse", "0.1sdtw"]:
                item[f"{m_1}_{criterion}"] = metrics[name]["means"][m_2][f"state_decoded_{criterion}"]

        item["duration"] = train_history[name]["means"]["duration"].sum()
        item["parameters"] = parameters[name]

        variations_data.append(item)

    df = pd.DataFrame(variations_data)

    _add_diverge_mask(df)
    _add_variables(df)

    return df

# Disjoint Groups


def variable_co_occurrence_graph(variations_df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    for i in range(len(variable_names)):
        for j in range(i+1, len(variable_names)):
            var_i = variable_names[i]
            var_j = variable_names[j]

            if variations_df[variations_df[var_i]][var_j].any():
                G.add_edge(var_i, var_j)

    return G


def variable_disjoint_graph(variable_co_occurrence_graph:  nx.Graph):
    return nx.complement(variable_co_occurrence_graph)


def disjoint_groups(variable_disjoint_graph:  nx.Graph) -> list[set[str]]:

    cliques = list(nx.find_cliques(variable_disjoint_graph))

    dg = []
    for c in cliques:
        if len(c) > 1:
            dg.append(set(c))

    return dg


def _get_disjoint_vars(variable: str,
                       disjoint_groups: list[set[str]]) -> set[str]:

    disjoint_vars = set()
    disjoint_vars.add(variable)
    for group in disjoint_groups:
        if variable in group:
            for group_v in group:
                disjoint_vars.add(group_v)
    return disjoint_vars

# Masked Percentage


def masked_percentage(variations_df: pd.DataFrame) -> float:
    return variations_df["mask"].sum()/variations_df


@datasaver()
def save_masked_percentage(masked_percentage: float, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, "masked_percentage.json")

    masked_percentage_dict = {"masked_percentage": masked_percentage}

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(masked_percentage_dict, file)

    return {"path": file_path}


# Task Metrics Distribution

def task_distribution_plots(variations_df: pd.DataFrame) -> dict[str, Figure]:
    figures = {}

    for metric in metric_names:
        items = []

        for i in range(len(variations_df)):
            if variations_df["mask"].iloc[i]:
                for task in task_names:
                    item = {}
                    item["task"] = format_name(metric).replace(" ", "\n")
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

# Diverge Probability


def divergence_probability(variations_df: pd.DataFrame) -> dict:
    unconditional_prob = 1-(variations_df["mask"].sum()/len(variations_df))

    conditional_diverge_prob = {}
    for v in variable_names:
        n_ndiverge = variations_df[variations_df[v]]["mask"].sum()
        n = len(variations_df[variations_df[v]])
        prob = 1 - (n_ndiverge/n).item()

        conditional_diverge_prob[v] = prob

    prob = {"unconditional": unconditional_prob,
            "conditional": conditional_diverge_prob}

    return prob


save_divergence_probability = function_variation({
    "metrics": source("divergence_probability"),
    "metrics_name": value("divergence_probability")},
    "save_divergence_probability")(save_metrics)


def filtered_divergence_probability(variations_df: pd.DataFrame) -> dict:
    to_ignore = ["AC_1"]

    var_mask = np.ones(len(variations_df))
    for iv in to_ignore:
        var_mask = np.bitwise_and(var_mask, np.bitwise_not(variations_df[iv]))

    n_ndiverge = variations_df["mask"][var_mask].sum()
    n = len(variations_df["mask"][var_mask])
    uncond_diverge_prob = 1 - (n_ndiverge/n).item()

    diverge_prob2 = {}
    for v in variable_names:
        if v not in to_ignore:
            var_mask = variations_df[v]
            for iv in to_ignore:
                var_mask = np.bitwise_and(
                    var_mask, np.bitwise_not(variations_df[iv]))

            n_ndiverge = variations_df[var_mask]["mask"].sum()
            n = len(variations_df[var_mask]).item()
            prob = 1 - (n_ndiverge/n)

            diverge_prob2[v] = prob

    prob = {"unconditional": uncond_diverge_prob,
            "conditional": diverge_prob2}

    return prob


save_filtered_divergence_probability = function_variation({
    "metrics": source("filtered_divergence_probability"),
    "metrics_name": value("filtered_divergence_probability")},
    "save_filtered_divergence_probability")(save_metrics)


def divergence_probability_plots(divergence_probability: dict) -> dict[str, Figure]:
    diverge_prob = divergence_probability["conditional"]
    uncond_diverge_prob = divergence_probability["unconditional"]

    fig = plt.figure(dpi=600)

    plt.bar(_format_variable_name(diverge_prob.keys()), 100 *
            np.array(list(diverge_prob.values())),
            label="P(Diverge|Variable)", color=palette[2])

    xlim = plt.xlim()

    plt.hlines([100*uncond_diverge_prob], xlim[0], xlim[1],
               colors=palette[0], label="P(Diverge)")

    plt.xlim(xlim)
    plt.ylim(0, 100)

    plt.xticks(rotation=45)
    plt.yticks([0, 20, 40, 60, 80, 100], [
               "0%", "20%", "40%", "60%", "80%", "100%"])

    plt.title("Divergence Probability")
    plt.ylabel("Probability")
    plt.xlabel("Variable")
    plt.legend()

    return {"divergence_probability": fig}


save_divergence_probability_plots = function_variation({
    "plots": source("divergence_probability_plots")},
    "save_divergence_probability_plots")(save_plots)


def filtered_divergence_probability_plots(filtered_divergence_probability: dict) -> dict[str, Figure]:
    diverge_prob = filtered_divergence_probability["conditional"]
    uncond_diverge_prob = filtered_divergence_probability["unconditional"]

    fig = plt.figure(dpi=600)

    plt.bar(_format_variable_name(diverge_prob.keys()), 100*np.array(list(diverge_prob.values())),
            label=r"$P(Diverge|Variable\cap\overline{AC}_1)$", color=palette[2])

    xlim = plt.xlim()
    plt.hlines([100*uncond_diverge_prob], xlim[0], xlim[1],
               colors=palette[0], label=r"$P(Diverge|\overline{AC}_1)$")
    plt.xlim(xlim)

    plt.ylim(0, 1)

    plt.xticks(rotation=45)

    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], [
        "0.0%", "0.2%", "0.4%", "0.6%", "0.8%", "1.0%"])

    plt.title("Divergence Probability\nGiven "+r"$\overline{AC}_1$")
    plt.ylabel("Probability")
    plt.xlabel("Variable")
    plt.legend()

    return {"filtered_divergence_probability": fig}


save_filtered_divergence_probability_plots = function_variation({"plots": source(
    "filtered_divergence_probability_plots")},
    "save_filtered_divergence_probability_plots")(save_plots)

# Impact tests


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


@extract_fields({"impact_test_df": pd.DataFrame, "impact_test_full_df": pd.DataFrame})
def impact_test_dfs(variations_df: pd.DataFrame,
                    disjoint_groups: list[set[str]]) -> dict[str, pd.DataFrame]:
    test_data = []
    test_full_data = []

    for var in variable_names:

        for task in metric_names+["duration"]:

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

                p_value, impact, _ = _joint_impact_test(
                    variations_df, disjoint_groups, var1, var2, f"{task}_mse")

                joint_impact[i, j] = impact
                joint_impact[j, i] = impact

                joint_impact_pvalue[i, j] = p_value
                joint_impact_pvalue[j, i] = p_value

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
    result = np.zeros_like(joint_impact)

    for dg in disjoint_groups:
        dg = list(dg)
        for i in range(len(dg)):
            for j in range(i+1, len(dg)):
                var_i = variable_names.index(dg[i])
                var_j = variable_names.index(dg[j])

                result[var_i, var_j] = 1
                result[var_j, var_i] = 1

    return result


def joint_impact_plots(joint_impact: dict,
                       disjoint_mask: np.ndarray):

    figures = {}
    for task in task_names:
        fig = plt.figure(dpi=600)

        impact = joint_impact["task"]["impact"]
        p_value = joint_impact["task"]["p_value"]

        range_limit = 2*np.std(impact)

        ji_img = plt.imshow(1000*impact,  cmap="bwr", vmin=-
                            1000*range_limit, vmax=1000*range_limit)
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
                    text = str(np.around(impact[i, j], decimals=3))
                    text = re.sub('0(?=[.])', '', text)
                    text = text[-2:]  # text[:2]+"\n"+text[2:]
                    text = str(int(np.sign(joint_impact[i, j]))*int(text))

                    fontsize = 10
                    if len(text) > 2:
                        fontsize = 8

                    color = "black"
                    if abs(int(text)) > 15:
                        color = "white"

                    ax.text(j, i,
                            text,
                            ha="center", va="center", color=color, fontsize=fontsize)

        plt.xlabel("Variable")
        plt.ylabel("Variable")
        plt.title("Variables Individual and Synergetic Impact\n" +
                  format_name(task)+" MSE " + r"$Impact\times1E-3$")

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
        joint_impact_filtered[disjoint_mask] = 0

        joint_impact_filtered = joint_impact_filtered[variables_mask][:variables_mask]
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
        individual_impact[individual_impact[task]["p_value"] >= 0.05] = 0
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
        if labels[i] in format_name(task_names):
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

    plt.title("Variable Impact on Duration")
    plt.ylabel("Time Impact [s] (Negative is Better)")
    plt.xlabel("Variable")

    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)

    return {"variable_impact_on_duration": fig}


save_duration_impact_plots = function_variation({
    "plots": source("duration_impact_plots")},
    "save_duration_impact_plots")(save_plots)

# Best models


def _get_best_theoretical_configuration(task: str,
                                        impact_test_df: pd.DataFrame,
                                        variable_disjoint_graph: nx.Graph,
                                        disjoint_groups: list[set[str]]) -> set[str]:

    final_vars = set()

    cc = list(nx.connected_components(variable_disjoint_graph))

    impacts = impact_test_df[impact_test_df["task"] == format_name(task)]
    impacts = impacts[["variable", "impact"]]
    impacts = impacts.set_index("variable")["impact"]
    impacts = impacts.to_dict()

    for group in cc:
        while len(group) != 0:
            min_impact = 0
            min_var = None

            for v in group:
                if impacts[v] < min_impact:
                    min_impact = impacts[v]
                    min_var = v

            if min_var is None:
                group.clear()
            else:
                final_vars.add(min_var)
                group.difference_update(
                    _get_disjoint_vars(min_var, disjoint_groups))

    return final_vars


def best_configurations(impact_test_df: pd.DataFrame,
                        variable_disjoint_graph: nx.Graph,
                        disjoint_groups: list[set[str]]) -> dict:
    best_configurations = {}
    best_configurations["theoretical"] = {}
    best_configurations["empirical"] = {}

    for task in task_names:
        best_configurations["theoretical"][task] = _get_best_theoretical_configuration(task,
                                                                                       impact_test_df,
                                                                                       variable_disjoint_graph,
                                                                                       disjoint_groups)

        empirical_mask = impact_test_df[impact_test_df["task"]
                                        == task][variable_names]
        empirical_mask = empirical_mask.iloc[0].to_numpy().astype(bool)

        best_configurations["empirical"][task] = variable_names[empirical_mask]

    return best_configurations


save_best_configurations = function_variation({
    "metrics": source("best_configurations"),
    "metrics_name": value("best_configurations")},
    "save_best_configurations")(save_metrics)


def best_models_df(variations_df: pd.DataFrame) -> pd.DataFrame:
    best_rows = {}
    for task in task_names:
        best_index = np.argmin(
            variations_df[variations_df["mask"]][f"{task}_mse"])
        best_row = variations_df[variations_df["mask"]].iloc[best_index]

        best_rows[task] = best_row

    df_best = pd.DataFrame(best_rows)
    df_best = df_best.T
    df_best = df_best.reset_index()

    df_best = df_best.rename(columns={"index": "task"})

    return df_best


save_best_models = function_variation({
    "metrics": source("best_models_df"),
    "metrics_name": value("best_models_df")},
    "save_best_models")(save_metrics)


def best_models_metrics_table(variations_df: pd.DataFrame,
                              best_models_df: pd.DataFrame,
                              best_configurations: dict) -> str:
    task_names_mse = []
    for task in task_names:
        task_names_mse.append(f"{task}_mse")

    columns_map = dict(zip(task_names_mse, format_name(task_names)))
    columns_map["task"] = "Best in"

    df_best_table = best_models_df.copy()

    df_best_table["task"] = df_best_table["task"].map(lambda x: format_name(x))

    df_best_table = df_best_table[["task"] +
                                  task_names_mse].rename(columns=columns_map)

    for task in best_configurations["theoretical"]:
        config_mask = np.ones(len(variations_df))

        for v in variable_names:
            if v in best_configurations["theoretical"][task]:
                config_mask = np.bitwise_and(config_mask, variations_df[v])
            else:
                config_mask = np.bitwise_and(
                    config_mask, np.bitwise_not(variations_df[v]))

        config_row = variations_df[config_mask]

        item = {"Type": "Theoretical",
                "Best in": format_name(task)}
        for t in metric_names:
            item[format_name(t)] = config_row[f"{t}_mse"].item()

        df_best_table = pd.concat([df_best_table, pd.DataFrame([item])])

    df_best_table.sort_values(
        "Best in", key=np.vectorize(format_name(task_names).index))

    return df_best_table.to_markdown(index=False)


def save_best_models_metrics_table(best_models_metrics_table: str,
                                   output_dir: str) -> dict:

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "best_models_metrics"+".json")

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(best_models_metrics_table)

    return {"path": file_path}
