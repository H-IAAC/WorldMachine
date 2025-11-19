
import networkx as nx
import pandas as pd

from ._shared import *


def variable_co_occurrence_graph(variations_df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    for i in range(len(variable_names)):
        for j in range(i+1, len(variable_names)):
            var_i = variable_names[i]
            var_j = variable_names[j]

            if variations_df[variations_df[var_i]][var_j].any():
                G.add_edge(var_i, var_j)

    return G


def variable_disjoint_graph(variable_co_occurrence_graph:  nx.Graph) -> nx.Graph:
    return nx.complement(variable_co_occurrence_graph)


def disjoint_groups(variable_disjoint_graph:  nx.Graph) -> list[set[str]]:

    cliques = list(nx.find_cliques(variable_disjoint_graph))

    dg = []
    for c in cliques:
        if len(c) > 1:
            dg.append(set(c))

    return dg
