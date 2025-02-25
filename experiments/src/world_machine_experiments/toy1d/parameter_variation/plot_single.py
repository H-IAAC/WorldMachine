import matplotlib.pyplot as plt
import torch
from hamilton.function_modifiers import source
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from world_machine import WorldMachine
from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.save_plots import save_plots
from world_machine_experiments.shared.train_plots import train_plots

save_toy1d_parameter_variation_plots = function_variation({"plots": source(
    "toy1d_parameter_variation_plots")}, "save_toy1d_parameter_variation_plots")(save_plots)
