import os

from hamilton.function_modifiers import datasaver
from matplotlib.figure import Figure


@datasaver()
def save_plots(plots:dict[str, Figure], output_dir:str) -> dict:

    plots_info = {}

    for name in plots:
        fig = plots[name]

        path = os.path.join(output_dir, name+".png")
        fig.savefig(path, facecolor="white", transparent=False)

        plots_info[name] = {"path":path}

    return plots_info