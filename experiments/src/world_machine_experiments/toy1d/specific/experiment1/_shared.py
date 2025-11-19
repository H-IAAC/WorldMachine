
import numpy as np

palette = ["#60BF60", "#D36767", "#6060BF"]

task_names = ["normal", "use_state", "prediction",
              "prediction_shallow", "prediction_local"]

metric_names = ["mse", "0.1sdtw"]

variable_names = np.array(["SB_1", "SB_2", "SM_1", "SM_2", "AC_1", "MD_1", "NA_1", "NA_2",
                           "RP_1", "RP_2", "RP_3", "RP_4",
                           "RF_1", "RF_2", "RF_3", "RF_4",
                           "LM_1"])


n_variable = len(variable_names)


def _format_variable_name(var: str | list):
    if not isinstance(var, str):
        return np.array([_format_variable_name(v) for v in var])

    return f"${var}$"


variable_names_f = _format_variable_name(variable_names)


def _get_disjoint_vars(variable: str,
                       disjoint_groups: list[set[str]]) -> set[str]:

    disjoint_vars = set()
    disjoint_vars.add(variable)
    for group in disjoint_groups:
        if variable in group:
            for group_v in group:
                disjoint_vars.add(group_v)
    return disjoint_vars
