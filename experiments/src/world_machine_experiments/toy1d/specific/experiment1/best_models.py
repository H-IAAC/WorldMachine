
import os

import networkx as nx
import numpy as np
import pandas as pd
from hamilton.function_modifiers import source, value

from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.acronyms import format_name
from world_machine_experiments.shared.save_metrics import save_metrics

from ._shared import *
from ._shared import _get_disjoint_vars


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
    "metrics_name": value("best_models")},
    "save_best_models")(save_metrics)


def best_configurations(impact_test_df: pd.DataFrame,
                        best_models_df: pd.DataFrame,
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

        empirical_mask = best_models_df[best_models_df["task"]
                                        == task][variable_names]
        empirical_mask = empirical_mask.iloc[0].to_numpy().astype(bool)

        best_configurations["empirical"][task] = variable_names[empirical_mask]

    return best_configurations


save_best_configurations = function_variation({
    "metrics": source("best_configurations"),
    "metrics_name": value("best_configurations")},
    "save_best_configurations")(save_metrics)


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

    df_best_table["Type"] = "Empirical"

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
        for t in task_names:
            item[format_name(t)] = config_row[f"{t}_mse"].item()

        df_best_table = pd.concat([df_best_table, pd.DataFrame([item])])

    df_best_table.sort_values(
        "Best in", key=np.vectorize(format_name(task_names).index))

    return df_best_table.to_markdown(index=False)


def save_best_models_metrics_table(best_models_metrics_table: str,
                                   output_dir: str) -> dict:

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "best_models_metrics"+".md")

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(best_models_metrics_table)

    return {"path": file_path}
