import os

import pydot
from hamilton.driver import Driver


def save_pipeline(driver: Driver,
                  final_vars: list,
                  file_name: str,
                  output_dir: str,
                  include_dot: bool = False) -> None:

    file_path = os.path.join(output_dir, file_name+".dot")

    driver.visualize_execution(final_vars=final_vars,
                               orient="TB",
                               deduplicate_inputs=True,
                               output_file_path=file_path,
                               bypass_validation=True)

    graph = load_graph(file_name, output_dir)
    clear_input_types(graph)

    save_graph(graph, file_name, output_dir, include_dot)

    if not include_dot:
        os.remove(os.path.join(output_dir, file_name+".dot"))


def load_graph(file_name: str, output_dir: str) -> pydot.Dot:
    graphs = pydot.graph_from_dot_file(
        os.path.join(output_dir, file_name+".dot"))

    return graphs[0]


def clear_input_types(graph: pydot.Dot) -> None:
    nodes: list[pydot.Node] = graph.get_node_list()
    labels: list[str] = []
    for node in nodes:
        name: str = node.get_name()
        if name.startswith("_") and name.endswith("_inputs"):

            label = node.get_label()

            labels.append(label)
            rows = label.split("</tr>")

            for i, row in enumerate(rows):
                columns = row.split("<td>")

                if len(columns) > 2:
                    columns = columns[:2]
                    rows[i] = "<td>".join(columns)

            label = "</tr>".join(rows)

            node.set_label(label)
            node.set_width("")


def save_graph(graph: pydot.Dot,
               file_name: str, output_dir: str,
               include_dot: bool = False) -> None:
    graph.write_svg(os.path.join(output_dir, file_name+".svg"))

    if include_dot:
        graph.write_dot(os.path.join(output_dir, file_name+".dot"))
