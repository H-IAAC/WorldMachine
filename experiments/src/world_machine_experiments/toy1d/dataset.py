import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from tensordict import TensorDict
from hamilton.function_modifiers import subdag, source

class Toy1dDataset(Dataset):
    def __init__(self, data:dict[str, np.ndarray], context_size:int,
                 return_state_dimensions:list[int]|None=None):
        super().__init__()

        self._data = data
        self._context_size = context_size
        self._return_dimensions = return_state_dimensions

        self._n_sequence = self._data["states"].shape[0]
        self._sequence_size = self._data["states"].shape[1]
        self._items_in_sequence = (self._sequence_size-1)//self._context_size
        self.size = self._n_sequence*int((self._sequence_size-1)/context_size)


    def __len__(self) -> int:
        return self.size
    
    
    def __getitem__(self, index):
        item_index = index // self._items_in_sequence
        item_seq_index = index % self._items_in_sequence

        start = item_seq_index*self._context_size
        end = start+self._context_size

        items = [TensorDict(), TensorDict()]
        for i in range(2):
            items[i]["state_decoded"] = torch.Tensor(self._data["states"][item_index, start+i:end+i])
            items[i]["state_control"] = torch.Tensor(self._data["state_controls"][item_index, start+i:end+i])
            items[i]["next_measurement"] = torch.Tensor(self._data["next_measurements"][item_index, start+i:end+i])

            if self._return_dimensions is not None:
                items[i]["state_decoded"] = items[i]["state_decoded"][:, self._return_dimensions]

        return TensorDict({"inputs":items[0], "targets":items[1]}, batch_size=[])
    


def toy1d_datasets(toy1d_data_splitted:dict[str, dict[str, np.ndarray]], context_size:int, 
             state_dimensions:list[int]|None=None) -> dict[str, Toy1dDataset]:

    datasets = {}
    for name in ["train", "val", "test"]:
        datasets[name] = Toy1dDataset(toy1d_data_splitted[name], 
                                      context_size=context_size, 
                                      return_state_dimensions=state_dimensions)
        

    return datasets