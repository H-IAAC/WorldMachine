import matplotlib.pyplot as plt
import torch
from hamilton.function_modifiers import source
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from world_machine import WorldMachine
from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.save_plots import save_plots
from world_machine_experiments.shared.train_plots import train_plots

multiple_toy1d_train_plots = function_variation({"train_history": source(
    "multiple_toy1d_consolidated_train_statistics")}, "multiple_toy1d_train_plots")(train_plots)
save_multiple_toy1d_train_plots = function_variation({"plots": source(
    "multiple_toy1d_train_plots")}, "save_multiple_toy1d_train_plots")(save_plots)
