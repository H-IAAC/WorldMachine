import os
import uuid

import torch

from world_machine.train import DatasetPassMode

from .train_stage import TrainStage


class EarlyStopper(TrainStage):
    def __init__(self):
        super().__init__(-1)

    def pre_train(self, model, criterions, train_criterions, device, optimizer):
        self._best_loss = torch.inf

        self._file_path = "best_model_"+str(uuid.uuid4())+".pth"
        while os.path.exists(self._file_path):
            self._file_path = "best_model_"+str(uuid.uuid4())+".pth"

        torch.save(model.state_dict(), self._file_path)

    def post_batch(self, model, losses, criterions, train_criterions, mode):
        if mode == DatasetPassMode.MODE_EVALUATE and losses["optimizer_loss"] < self._best_loss:
            self._best_loss = losses["optimizer_loss"]
            torch.save(model.state_dict(), self._file_path)

    def post_train(self, model, criterions, train_criterions, optimizer):
        model.load_state_dict(torch.load(self._file_path, weights_only=True))

        os.remove(self._file_path)
