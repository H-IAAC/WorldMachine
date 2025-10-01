import os

from hamilton.function_modifiers import datasaver
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


@datasaver()
def save_plots(plots: dict[str, Figure], output_dir: str) -> dict:

    os.makedirs(output_dir, exist_ok=True)

    plots_info = {}

    for name in plots:
        fig = plots[name]

        path = os.path.join(output_dir, name+".png")
        fig.savefig(path, facecolor="white",
                    transparent=False,  bbox_inches="tight")

        plots_info[name] = {"path": path}

        plt.close(fig)

    return plots_info
