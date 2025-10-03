from typing import Callable

from .train_stage import TrainStage


class LambdaStage(TrainStage):

    def __init__(self, execution_order: float,
                 pre_train: Callable | None = None,
                 pre_batch: Callable | None = None,
                 pre_segment: Callable | None = None,
                 pre_forward: Callable | None = None,
                 forward: Callable | None = None,
                 post_forward: Callable | None = None,
                 post_segment: Callable | None = None,
                 optimize: Callable | None = None,
                 post_batch: Callable | None = None,
                 post_train: Callable | None = None):
        super().__init__(execution_order)

        self._functions = {}
        self._functions["pre_train"] = pre_train
        self._functions["pre_batch"] = pre_batch
        self._functions["pre_segment"] = pre_segment
        self._functions["pre_forward"] = pre_forward
        self._functions["forward"] = forward
        self._functions["post_forward"] = post_forward
        self._functions["post_segment"] = post_segment
        self._functions["optimize"] = optimize
        self._functions["post_batch"] = post_batch
        self._functions["post_train"] = post_train

        if forward is None:
            self._with_forward = False

    def pre_train(self, model, criterions, train_criterions, device, optimizer):
        if self._functions["pre_train"] is not None:
            self._functions["pre_train"](
                model, criterions, train_criterions, device, optimizer)

    def pre_batch(self, model, mode, criterions, optimizer, device, losses, train_criterions):
        if self._functions["pre_batch"] is not None:
            self._functions["pre_batch"](
                model, mode, criterions, optimizer, device, losses, train_criterions)

    def pre_segment(self, itens, losses, batch_size, seq_len, epoch_index, device, state_size, mode, model):
        if self._functions["pre_segment"] is not None:
            self._functions["pre_segment"](
                itens, losses, batch_size, seq_len, epoch_index, device, state_size, mode, model)

    def pre_forward(self, item_index, itens, mode, batch_size, device, epoch_index):
        if self._functions["pre_forward"] is not None:
            self._functions["pre_forward"](
                item_index, itens, mode, batch_size, device, epoch_index)

    def forward(self, model, segment, mode):
        if self._functions["forward"] is not None:
            self._functions["forward"](model, segment, mode)

    def post_forward(self, item_index, itens, dataset, losses, mode):
        if self._functions["post_forward"] is not None:
            self._functions["post_forward"](
                item_index, itens, dataset, losses, mode)

    def post_segment(self, itens, losses, dataset, epoch_index, criterions, mode, device, train_criterions):
        if self._functions["post_segment"] is not None:
            self._functions["post_segment"](
                itens, losses, dataset, epoch_index, criterions, mode, device, train_criterions)

    def optimize(self, model, optimizer, batch_index, n_batch, losses, mode):
        if self._functions["optimize"] is not None:
            self._functions["optimize"](
                model, optimizer, batch_index, n_batch, losses, mode)

    def post_batch(self, model, losses, criterions, train_criterions, mode):
        if self._functions["post_batch"] is not None:
            self._functions["post_batch"](
                model, losses, criterions, train_criterions, mode)

    def post_train(self, model, criterions, train_criterions, optimizer):
        if self._functions["post_train"] is not None:
            self._functions["post_train"](
                model, criterions, train_criterions, optimizer)
