import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from hamilton.function_modifiers import source, value
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from world_machine import WorldMachine
from world_machine.train import Trainer
from world_machine.train.trainer import MODE_EVALUATE
from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.save_plots import save_plots
from world_machine_experiments.shared.save_train_history import (
    save_train_history)


def toy1d_mask_sensorial_metrics(toy1d_model_trained: WorldMachine,
                                 toy1d_dataloaders: dict[str, DataLoader],
                                 toy1d_trainer: Trainer,
                                 discover_state: bool = False) -> dict[str, np.ndarray]:

    mask_sensorial_percentage = np.array([0, 0.25, 0.5, 0.75, 1.0])
    state_size = toy1d_model_trained._state_size
    device = next(iter(toy1d_model_trained.parameters())).device
    mse = torch.nn.MSELoss()

    batch_size = toy1d_dataloaders[next(iter(toy1d_dataloaders))].batch_size
    seq_len = toy1d_model_trained._max_context_size

    metrics = {"mask_sensorial_percentage": mask_sensorial_percentage}

    for split in ["train", "val"]:
        loader = toy1d_dataloaders[split]

        metrics["mask_sensorial_" +
                split] = np.empty_like(mask_sensorial_percentage)

        for i, pct in enumerate(mask_sensorial_percentage):

            toy1d_trainer._mask_sensorial_data = pct

            if discover_state:
                toy1d_model_trained.eval()

                with torch.no_grad():
                    total_loss = torch.tensor(
                        0, dtype=torch.float32, device=device)
                    n = 0
                    for item in tqdm.tqdm(loader, desc="Autoregressive Mask Sensorial Metrics"):
                        inputs: torch.Tensor = item["inputs"].to(device)
                        targets: torch.Tensor = item["targets"]["state_decoded"].to(
                            device)

                        state = torch.rand(
                            (batch_size, seq_len, state_size), device=device)
                        state = (2*state)-1

                        sensorial_masks = toy1d_trainer._generate_sensorial_masks(
                            inputs, MODE_EVALUATE, True, device, batch_size, seq_len)

                        for i in range(seq_len):
                            logits = toy1d_model_trained(
                                state=state,
                                sensorial_data=inputs, sensorial_masks=sensorial_masks)

                            if i != seq_len-1:
                                state[:, i+1] = logits["state"][:, i]

                        loss = mse(targets[:, :, 0],
                                   logits["state_decoded"][:, :, 0])
                        total_loss += loss * targets.size(0)

                        n += targets.size(0)

                    total_loss = total_loss.cpu().item()
                    total_loss /= n

                    metrics["mask_sensorial_" +
                            split][i] = total_loss

            else:
                losses = toy1d_trainer._compute_loss_and_optimize(
                    toy1d_model_trained, loader, MODE_EVALUATE, None, 0, True)

                metrics["mask_sensorial_" +
                        split][i] = losses["optimizer_loss"].cpu().item()

    return metrics


def toy1d_masks_sensorial_plots(toy1d_mask_sensorial_metrics: dict[str, np.ndarray], y_scale: str = "log") -> dict[str, Figure]:

    fig = plt.figure(dpi=300)

    plot_args = {"fmt": "o-", "capsize": 5.0, "markersize": 4}

    for name in ["train", "val"]:
        if "mask_sensorial_train_std" in toy1d_mask_sensorial_metrics:
            plt.errorbar(toy1d_mask_sensorial_metrics["mask_sensorial_percentage"],
                         toy1d_mask_sensorial_metrics["mask_sensorial_" + name],
                         toy1d_mask_sensorial_metrics["mask_sensorial_"+name+"_std"],
                         label=name, **plot_args)
        else:
            plt.plot(toy1d_mask_sensorial_metrics["mask_sensorial_percentage"],
                     toy1d_mask_sensorial_metrics["mask_sensorial_" + name],
                     label=name, **plot_args)

    plt.title("Mask Sensorial Loss")
    plt.xlabel("Masking Percentage")
    plt.ylabel("Loss")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

    plt.yscale(y_scale)

    plt.close()

    figures = {"mask_sensorial": fig}

    return figures


save_toy1d_masks_sensorial_plot = function_variation({"plots": source(
    "toy1d_masks_sensorial_plots")}, "save_toy1d_masks_sensorial_plot")(save_plots)

save_toy1d_mask_sensorial_metrics = function_variation({"train_history": source(
    "toy1d_mask_sensorial_metrics"), "history_name": value("mask_sensorial_metrics")}, "save_toy1d_mask_sensorial_metrics")(save_train_history)
