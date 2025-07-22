from torch.nn import Module


class CriterionSet:
    def __init__(self):
        self.criterions: dict[str, dict[str, Module]] = {}
        self.criterions["state_decoded"] = {}

        self.train_criterions: dict[str, dict[str, float]] = {}
        self.train_criterions["state_decoded"] = {}

    def add_decoded_state_criterion(self, name: str,
                                    criterion: Module,
                                    train: bool = False,
                                    weight: float = 1.0) -> None:
        self.criterions["state_decoded"][name] = criterion

        if train:
            self.train_criterions["state_decoded"][name] = weight

    def add_sensorial_criterion(self, name: str, sensorial_dimension: str, criterion: Module, train: bool = False,
                                weight: float = 1.0) -> None:
        if sensorial_dimension not in self.criterions:
            self.criterions[sensorial_dimension] = {}
        if sensorial_dimension not in self.train_criterions:
            self.train_criterions[sensorial_dimension] = {}

        self.criterions[sensorial_dimension][name] = criterion

        if train:
            self.train_criterions[sensorial_dimension][name] = weight
