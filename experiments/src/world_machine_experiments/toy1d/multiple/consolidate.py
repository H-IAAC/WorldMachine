from hamilton.function_modifiers import source, value

from world_machine_experiments.shared import function_variation
from world_machine_experiments.shared.save_train_history import (
    save_train_history)
from world_machine_experiments.shared.statistics import (
    consolidated_train_statistics)

save_multiple_toy1d_consolidated_train_statistics = function_variation({"train_history": source(
    "multiple_toy1d_consolidated_train_statistics"), "history_name": value("toy1d_train_history")}, "save_multiple_toy1d_consolidated_train_statistics")(save_train_history)


multiple_toy1d_consolidated_train_statistics = function_variation({"training_infos": source(
    "multiple_toy1d_trainings_info")}, "multiple_toy1d_consolidated_train_statistics")(consolidated_train_statistics)
