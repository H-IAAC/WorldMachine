import matplotlib.pyplot as plt
import torch
from hamilton.function_modifiers import source
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from world_machine import WorldMachine
from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.save_plots import save_plots
from world_machine_experiments.shared.train_plots import train_plots


def toy1d_prediction_plots(toy1d_model_trained: WorldMachine, toy1d_dataloaders: dict[str, DataLoader]) -> dict[str, Figure]:
    device = next(iter(toy1d_model_trained.parameters())).device
    toy1d_model_trained.eval()

    figures = {}

    for name in ["train", "val"]:
        item = next(iter(toy1d_dataloaders[name]))

        inputs: torch.Tensor = item["inputs"].to(device)
        targets: torch.Tensor = item["targets"]["state_decoded"]

        with torch.no_grad():
            logits: torch.Tensor = toy1d_model_trained(
                state_decoded=inputs["state_decoded"], sensory_data=inputs)

        logits = logits.cpu().numpy()

        axis = 0
        fig, axs = plt.subplots(4, 8, dpi=300)

        for i in range(32):
            row = i // 8
            column = i % 8

            axs[row, column].plot(targets[i, :, axis], label="Ground Truth")
            axs[row, column].plot(logits["state_decoded"]
                                  [i][:, axis], label="Prediction")

            axs[row, column].set_xticks([])
            axs[row, column].set_yticks([])

        plt.suptitle("Prediction Sample - "+name.capitalize())
        plt.legend(bbox_to_anchor=(2.5, 4.5), loc='upper right')

        plt.close()

        figures[name] = fig

    return figures


toy1d_train_plots = function_variation(
    {"train_history": source("toy1d_train_history")}, "toy1d_train_plots")(train_plots)

save_toy1d_train_plots = function_variation(
    {"plots": source("toy1d_train_plots")}, "save_toy1d_train_plots")(save_plots)

save_toy1d_prediction_plots = function_variation({"plots": source(
    "toy1d_prediction_plots")}, "save_toy1d_prediction_plots")(save_plots)
