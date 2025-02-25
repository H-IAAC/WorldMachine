import numpy as np


def consolidated_train_statistics(training_infos: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:

    consolidated = {}
    for name in training_infos[0]:
        entry_from_all = []
        for i in range(len(training_infos)):
            entry_from_all.append(training_infos[i][name])

        consolidated[name] = np.mean(entry_from_all, axis=0)
        consolidated[name+"_std"] = np.std(entry_from_all, axis=0)

    return consolidated
