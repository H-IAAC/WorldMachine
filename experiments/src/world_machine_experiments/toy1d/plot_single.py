import os

import torch
from torch.utils.data import DataLoader
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from hamilton.function_modifiers import datasaver, parameterize, source

from world_machine import WorldMachine

acronyms = ["MSE"]

@parameterize(multiple_toy1d_train_plots={"toy1d_train_history":source("multiple_toy1d_consolidated_train_statistics")},
              toy1d_train_plots={})
def toy1d_train_plots(toy1d_train_history:dict[str, np.ndarray],
                      log_y_axis:bool=True) -> dict[str, Figure]:
    names : set[str] = set()

    with_std = False

    for name in toy1d_train_history:
        if name.endswith("_std"):
            with_std = True
        
        names.add(name.removesuffix("_std").removesuffix("_train").removesuffix("_val"))

    names.remove("duration")

    n_epoch = len(toy1d_train_history["duration"])
    epochs = range(1, n_epoch+1)

    figures = {}
    for name in names:
        fig = plt.figure(dpi=300)

        train_hist = toy1d_train_history[name+"_train"]
        val_hist = toy1d_train_history[name+"_val"]


        if with_std:
            train_hist_std = toy1d_train_history[name+"_train_std"]
            val_hist_std = toy1d_train_history[name+"_val_std"]
            
            plot_args = {"fmt":"o-", "capsize":5.0, "markersize":4}

            plt.errorbar(epochs, train_hist, train_hist_std, label="Train", **plot_args)
            plt.errorbar(epochs, val_hist, val_hist_std, label="Validation", **plot_args)
        else:
            plt.plot(epochs, train_hist, "o-", label="Train")
            plt.plot(epochs, val_hist, "o-", label="Validation")

        name_format = name.replace("_", " ").title()

        for acro in acronyms:
            name_format = name_format.replace(acro.capitalize(), acro)

        plt.title(name_format)
        plt.xlabel("Epochs")
        plt.ylabel("Metric")
        plt.legend()

        if log_y_axis:
            plt.yscale("log")

        plt.close()

        figures[name] = fig

    return figures
        




def toy1d_prediction_plots(toy1d_model_trained:WorldMachine, toy1d_dataloaders:dict[str, DataLoader]) -> dict[str, Figure]:
    device = next(iter(toy1d_model_trained.parameters())).device
    toy1d_model_trained.eval()

    figures = {}

    for name in toy1d_dataloaders:
        item = next(iter(toy1d_dataloaders[name]))

        inputs : torch.Tensor = item["inputs"].to(device)
        targets : torch.Tensor = item["targets"]["state_decoded"]

        with torch.no_grad():
            logits : torch.Tensor = toy1d_model_trained(inputs["state_decoded"], inputs)

        logits = logits.cpu().numpy()

        axis = 0
        fig, axs = plt.subplots(4, 8, dpi=300)

        for i in range(32):
            row = i // 8
            column = i % 8

            axs[row, column].plot(targets[i,:,axis], label="Ground Truth")
            axs[row, column].plot(logits["state_decoded"][i][:,axis], label="Prediction")

            axs[row, column].set_xticks([])
            axs[row, column].set_yticks([])
        
        plt.suptitle("Prediction Sample - "+name.capitalize())
        plt.legend(bbox_to_anchor=(2.5, 4.5), loc='upper right')

        plt.close()

        figures[name] = fig

    return figures
        

@parameterize(save_toy1d_train_plots={"plots":source("toy1d_train_plots")}, 
              save_toy1d_prediction_plots={"plots":source("toy1d_prediction_plots")},
              save_multiple_toy1d_train_plots={"plots":source("multiple_toy1d_train_plots")})
@datasaver()
def save_plots(plots:dict[str, Figure], output_dir:str) -> dict:

    plots_info = {}

    for name in plots:
        fig = plots[name]

        path = os.path.join(output_dir, name+".png")
        fig.savefig(path, facecolor="white", transparent=False)

        plots_info[name] = {"path":path}

    return plots_info