import abc
import atexit
import os
from collections import deque

import torch
from tensordict import LazyStackedTensorDict, stack
from torch.utils.data import DataLoader


class WorldMachineDataLoader(DataLoader):
    def __init__(self, dataset,
                 batch_size=1,
                 shuffle=None,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 pin_memory=False,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None,
                 multiprocessing_context=None,
                 generator=None,
                 *,
                 prefetch_factor=None,
                 persistent_workers=False,
                 pin_memory_device=""):
        def collate_fn(x): return stack(list(x))

        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn,
                         multiprocessing_context, generator, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, pin_memory_device=pin_memory_device)
