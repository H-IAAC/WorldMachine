import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from hamilton.function_modifiers import datasaver, extract_fields, source
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from world_machine import WorldMachine
from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.save_metrics import save_metrics
from world_machine_experiments.shared.save_plots import save_plots


@extract_fields(fields={"toy1d_autoregressive_losses": dict[str, float],
                        "toy1d_autoregressive_states": dict[str, torch.Tensor],
                        "toy1d_autoregressive_states_decoded": dict[str, torch.Tensor]})
def toy1d_autoregressive_info(toy1d_model_trained: WorldMachine,
                              toy1d_dataloaders: dict[str, DataLoader]) -> dict[str, dict[str, float] | dict[str, torch.Tensor]]:
    mse = torch.nn.MSELoss()
    device = next(iter(toy1d_model_trained.parameters())).device

    toy1d_model_trained.eval()

    losses: dict[str, float] = {}
    states: dict[str, torch.Tensor] = {}
    state_decoded: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for dataset_name in ["train", "val"]:

            loader = toy1d_dataloaders[dataset_name]
            states[dataset_name] = torch.empty_like(
                loader.dataset._states, device="cpu")
            state_decoded[dataset_name] = torch.empty(
                (len(loader.dataset), toy1d_model_trained._max_context_size, 3))

            total_loss = torch.tensor(0, dtype=torch.float32, device=device)
            n = 0
            for item in tqdm.tqdm(loader, desc="Autoregressive inference"):
                inputs: torch.Tensor = item["inputs"].to(device)
                targets: torch.Tensor = item["targets"]["state_decoded"].to(
                    device)

                if "state_decoded" in inputs:
                    batch_size = inputs["state_decoded"].shape[0]
                    seq_len = inputs["state_decoded"].shape[1]

                state_size = toy1d_model_trained._state_size

                state = torch.rand(
                    (batch_size, seq_len, state_size), device=device)
                state = (2*state)-1

                sensorial_masks = None

                for i in range(seq_len):
                    logits = toy1d_model_trained(
                        state=state,  # state.clone(),
                        sensorial_data=inputs, sensorial_masks=sensorial_masks)

                    if i != seq_len-1:
                        state[:, i+1] = logits["state"][:, i]

                loss = mse(targets[:, :, 0], logits["state_decoded"][:, :, 0])
                total_loss += loss * targets.size(0)

                n += targets.size(0)

                indexes = item["index"].cpu()

                states[dataset_name][indexes] = state.cpu()
                state_decoded[dataset_name][indexes] = logits["state_decoded"].cpu()

            total_loss = total_loss.item()
            total_loss /= n

            losses[dataset_name] = total_loss

    output = {"toy1d_autoregressive_losses": losses,
              "toy1d_autoregressive_states": states,
              "toy1d_autoregressive_states_decoded": state_decoded}

    return output


def toy1d_autoregressive_state_plots(toy1d_autoregressive_states: dict[str, torch.Tensor], toy1d_dataloaders: dict[str, DataLoader]) -> dict[str, Figure]:
    figures = {}

    for dataset_name in toy1d_autoregressive_states:
        fig, axs = plt.subplots(4, 6, dpi=600, figsize=(12, 4.8))
        fig.subplots_adjust(wspace=.5)

        parallel_states: torch.Tensor = toy1d_dataloaders[dataset_name].dataset._states

        for i in range(24):
            row = i // 6
            column = i % 6

            axs[row, column].plot(
                parallel_states[row, :, column].cpu(), color="red", label="Parallel")
            axs[row, column].plot(toy1d_autoregressive_states[dataset_name][row, :, column].cpu(),
                                  color="blue", label="Autoregressive")
            axs[row, column].set_xticks([])
            # axs[row, column].set_yticks([])

        for i in range(6):
            axs[0][i].set_title(f"Dim {i}")

        for i in range(4):
            axs[i][0].set_ylabel(f"Element {i}", size="large")

        plt.suptitle("State Sample - Train")
        plt.legend(bbox_to_anchor=(2.5, 4.5), loc='upper right')
        plt.close()

        figures[f"AutoregressiveState_{dataset_name}"] = fig

    return figures


def toy1d_autoregressive_positional_encoder_plots(toy1d_model_trained: WorldMachine,
                                                  toy1d_autoregressive_states: dict[str, torch.Tensor],
                                                  toy1d_dataloaders: dict[str, DataLoader]) -> dict[str, Figure]:
    pe = toy1d_model_trained._positional_encoder().cpu()[:, 0]

    seq_len = pe.shape[0]

    pe_outoffase = -np.sin(seq_len*np.linspace(0, 1, seq_len) + (np.pi/2))

    parallel_states: torch.Tensor = toy1d_dataloaders["train"].dataset._states

    fig, axs = plt.subplots(1, 6, dpi=600, figsize=(30, 4))

    for column in range(6):

        axs[column].plot(pe, label="Positional Encoder")
        axs[column].plot(pe_outoffase, label="Out fase positional encoder")

        axs[column].plot(parallel_states[0, :, column].cpu(),
                         color="red", label="Parallel")
        axs[column].plot(toy1d_autoregressive_states["train"][0, :, column].cpu(),
                         color="blue", label="Autoregressive")

        axs[column].set_xlim(0, 50)

        axs[column].set_title(f"Dim {column}")

    plt.legend(bbox_to_anchor=(1.8, 1.0), loc='upper right')

    plt.suptitle("Positional Encoding x State - Sample")

    plt.close()

    output = {"PositionalEncodingxStateSample": fig}

    return output


def toy1d_autoregressive_state_decoded_plots(toy1d_model_trained: WorldMachine, toy1d_autoregressive_states_decoded: dict[str, torch.Tensor], toy1d_dataloaders: dict[str, DataLoader]) -> dict[str, Figure]:
    device = next(iter(toy1d_model_trained.parameters())).device

    item = next(iter(toy1d_dataloaders["train"]))

    inputs: torch.Tensor = item["inputs"].to(device)
    state_decoded_target: torch.Tensor = item["targets"]["state_decoded"].cpu()

    state = inputs["state"]
    sensorial_masks = None

    with torch.no_grad():
        logits_orig = toy1d_model_trained(
            state=state, sensorial_data=inputs, sensorial_masks=sensorial_masks)

    indexes = item["index"].cpu()

    state_decoded_autoregressive = toy1d_autoregressive_states_decoded["train"][indexes].cpu(
    )
    state_decoded_parallel: torch.Tensor = logits_orig["state_decoded"].cpu()

    fig, axs = plt.subplots(4, 8, dpi=600, figsize=(12, 4.8))

    for i in range(32):
        row = i // 8
        column = i % 8

        axs[row, column].plot(
            state_decoded_target[i, :, 0].cpu(), label="Target")
        axs[row, column].plot(
            state_decoded_parallel[i, :, 0].cpu(), label="Prediction Parallel")
        axs[row, column].plot(
            state_decoded_autoregressive[i, :, 0].cpu(), label="Prediction Autoregressive")

        axs[row, column].set_xticks([])
        axs[row, column].set_yticks([])

    plt.suptitle("Prediction Sample - Parallel x Autoregressive")
    plt.legend(bbox_to_anchor=(3.5, 4.5), loc='upper right')

    plt.close()

    output = {"Prediction-AutoregressivexParallel": fig}

    return output


save_toy1d_autoregressive_state_plots = function_variation(
    {"plots": source("toy1d_autoregressive_state_plots")}, "save_toy1d_autoregressive_state_plots")(save_plots)

save_toy1d_autoregressive_positional_encoder_plots = function_variation(
    {"plots": source("toy1d_autoregressive_positional_encoder_plots")}, "save_toy1d_autoregressive_positional_encoder_plots")(save_plots)

save_toy1d_autoregressive_state_decoded_plots = function_variation(
    {"plots": source("toy1d_autoregressive_state_decoded_plots")}, "save_toy1d_autoregressive_state_decoded_plots")(save_plots)


def toy1d_autoregressive_metrics(toy1d_autoregressive_losses: dict[str, float],
                                 toy1d_train_history: dict[str, np.ndarray]) -> dict:

    parallel_metrics = {}

    for name in ["train", "val"]:
        parallel_metrics[name] = toy1d_train_history[
            f"state_decoded_mse_first_{name}"][-1]

    relation = {}

    for name in ["train", "val"]:
        relation[name] = toy1d_autoregressive_losses[name] / \
            parallel_metrics[name]

    total_metrics = {}
    total_metrics["autoregressive"] = toy1d_autoregressive_losses
    total_metrics["parallel"] = parallel_metrics
    total_metrics["proportion"] = relation

    return total_metrics


save_toy1d_autoregressive_metrics = function_variation({"metrics": source(
    "toy1d_autoregressive_metrics")}, "save_toy1d_autoregressive_metrics")(save_metrics)
