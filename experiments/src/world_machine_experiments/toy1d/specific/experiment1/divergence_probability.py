

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hamilton.function_modifiers import source, value
from matplotlib.figure import Figure

from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.save_metrics import save_metrics
from world_machine_experiments.shared.save_plots import save_plots

from ._shared import *
from ._shared import _format_variable_name


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
            n = len(variations_df[var_mask])
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
