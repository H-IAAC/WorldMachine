from typing import Callable
import time

import torch
from torch.nn import Module
import numpy as np
from numpy.typing import ArrayLike
import tqdm
from tensordict import TensorDict
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wanbd = None

from world_machine.world_machine import WorldMachine

MODE_TRAIN = 0
MODE_EVALUATE = 1

class Trainer:
    def __init__(self, discover_state:bool):

        if discover_state:
            raise NotImplementedError()

        self.discover_state = discover_state

        self._criterions : dict[str, dict[str, Module]] = {}
        self._criterions["state_decoded"] = {}

        self._train_criterions : dict[str, dict[str, float]] = {}
        self._train_criterions["state_decoded"] = {}

    def add_decoded_state_criterion(self, name:str, 
                                    criterion:Module, 
                                    train:bool=False,
                                    weight:float=1.0) -> None:
        self._criterions["state_decoded"][name] = criterion

        if train:
            self._train_criterions["state_decoded"][name] = weight

    def add_sensorial_criterion(self, name:str, sensorial_dimension:str, criterion:Module, train:bool=False,
                                weight:float=1.0) -> None:
        if sensorial_dimension not in self._criterions:
            self._criterions[sensorial_dimension] = {}
        
        self._criterions[sensorial_dimension][name] = criterion

        if train:
            if sensorial_dimension not in self._train_criterions:
                self._train_criterions[sensorial_dimension] = set()

            self._train_criterions[sensorial_dimension][name] = weight

    def __call__(self, wm:WorldMachine, 
                 dataloaders:dict[str, DataLoader],
                 optimizer:torch.optim.Optimizer,
                 n_epoch:int,
                 accumulation_steps:int=1) ->  dict[str, np.ndarray|dict[str, np.ndarray]|dict[str, dict[str, np.ndarray]]]:
        
        return self._train(wm, dataloaders, optimizer, n_epoch, accumulation_steps, False, None)


    def _compute_loss_and_optimize(self,
                                  model:WorldMachine, 
                                loader:DataLoader, 
                                mode:int = MODE_EVALUATE, 
                                optimizer:torch.optim.Optimizer|None=None, 
                                accumulation_steps:int|None = None) -> dict[str, torch.Tensor]:
        """
        Computes the loss from a model across a dataset.

        If in train mode also runs optimizer steps.

        Args:
            model (torch.nn.Module): model to evaluate.
            loader (DataLoader): dataset.
            mode (int): mode of the computation. 
                        If MODE_EVALUATE, computes without gradient, in eval mode and detachs loss.
                        If MODE_TRAIN, computes with gradient and in train mode.
                        Default is MODE_EVALUATE.
            optimizer (torch.optim.Optimizer, optional): optimizer to use in the train mode.

        Returns:
            torch.Tensor: resulting loss.
        """
        if accumulation_steps is None:
            accumulation_steps = 1

        original_grad_state = torch.is_grad_enabled()
        original_model_state = model.training

        device = next(iter(model.parameters())).device

        if mode == MODE_EVALUATE:
            model.eval()
            torch.set_grad_enabled(False)
        elif mode == MODE_TRAIN:
            model.train()
            torch.set_grad_enabled(True)
            optimizer.zero_grad()
        else:
            raise ValueError(f"Unknown mode: {mode}.")

        batch_index = 0

        total_loss : dict[str, dict[str, torch.Tensor]|torch.Tensor] = {}
        for dimension in self._criterions:
            total_loss[dimension] = {}
            for criterion_name in self._criterions[dimension]:
                total_loss[dimension][criterion_name] = torch.tensor(0, dtype=torch.float32, device=device)

        total_loss["optimizer_loss"] = torch.tensor(0, dtype=torch.float32, device=device)

        n = 0
        for item in tqdm.tqdm(loader):
            item = item.to(device)
            inputs : torch.Tensor = item["inputs"]
            targets : torch.Tensor = item["targets"]
            
            logits : TensorDict = model(inputs["state_decoded"], inputs)

            losses : dict[str, dict[str, torch.Tensor]|torch.Tensor] = {}


            for dimension in self._criterions:
                losses[dimension] = {}
                for criterion_name in self._criterions[dimension]:
                    losses[dimension][criterion_name] = self._criterions[dimension][criterion_name](logits[dimension], targets[dimension])

                    total_loss[dimension][criterion_name] += losses[dimension][criterion_name]*targets.size(0)

            optimizer_loss = torch.tensor(0, dtype=torch.float32, device=device)
            for dimension in self._train_criterions:
                for criterion_name in self._train_criterions[dimension]:
                    optimizer_loss += losses[dimension][criterion_name] * self._train_criterions[dimension][criterion_name]

            losses["optimizer_loss"] = optimizer_loss
            total_loss["optimizer_loss"] += losses["optimizer_loss"] * targets.size(0)

            n += targets.size(0)

            if mode == MODE_TRAIN:
                optimizer_loss /= accumulation_steps
                optimizer_loss.backward()

                if ((batch_index+1) % accumulation_steps == 0) or (batch_index+1 == len(loader)):
                    optimizer.step()
                    optimizer.zero_grad()

            batch_index += 1

        for dimension in total_loss:
            if dimension == "optimizer_loss":
                total_loss[dimension] /= n
                total_loss[dimension] = total_loss[dimension].detach()
            else:
                for criterion_name in total_loss[dimension]:
                    total_loss[dimension][criterion_name] /= n 
                    total_loss[dimension][criterion_name] = total_loss[dimension][criterion_name].detach()

        

        result = {}
        for dimension in total_loss:
            if dimension == "optimizer_loss":
                result[dimension] = total_loss[dimension]
            else:
                for criterion_name in total_loss[dimension]:
                    result[f"{dimension}_{criterion_name}"] = total_loss[dimension][criterion_name]

        #Return original state
        torch.set_grad_enabled(original_grad_state)
        
        if original_model_state:
            model.train()
        else:
            model.eval()

        return result
    
    
    def _train(self, wm:WorldMachine, 
                dataloaders:dict[str, DataLoader],
                optimizer:torch.optim.Optimizer,
                n_epoch:int,
                accumulation_steps:int,
                use_wandb:bool=False, 
                early_stop:Callable[[float], bool]|None=None) ->  dict[str, np.ndarray]:
    
        hist : dict[str, np.ndarray|dict[str, np.ndarray]|dict[str, dict[str, np.ndarray]]] = {}
        for dimension in self._criterions:
            hist[dimension] = {}
            for criterion_name in self._criterions[dimension]:
                hist[dimension][criterion_name] = {"train":np.empty(n_epoch),
                                                   "val":np.empty(n_epoch)}
                
        hist["optimizer_loss"] = {"train":np.empty(n_epoch),
                            "val":np.empty(n_epoch)}

        hist["duration"] = np.empty(n_epoch)

        loss_val = self._compute_loss_and_optimize(wm, dataloaders["val"], MODE_EVALUATE)
            
        print("VAL ", end="")
        print_info(loss_val["optimizer_loss"], -1, n_epoch)

        for epoch in range(n_epoch):
            start_time = time.time() 

            loss_train = self._compute_loss_and_optimize(wm, dataloaders["train"], MODE_TRAIN, optimizer, accumulation_steps)

            end_time = time.time() 
            
            epoch_duration = end_time - start_time 


            print_info(loss_train["optimizer_loss"], epoch, n_epoch, epoch_duration)
            
            #Validation stats
            loss_val = self._compute_loss_and_optimize(wm, dataloaders["val"], MODE_EVALUATE)
            
            print("VAL ", end="")
            print_info(loss_val["optimizer_loss"], epoch, n_epoch)

            #Save history and log
            log : dict[str, float] = {}

            for dimension in self._criterions:
                for criterion_name in self._criterions[dimension]:
                    hist[dimension][criterion_name]["train"][epoch] = loss_train[f"{dimension}_{criterion_name}"].item()
                    hist[dimension][criterion_name]["val"][epoch] = loss_val[f"{dimension}_{criterion_name}"].item()

                    log[f"loss_train_{dimension}_{criterion_name}"] = loss_train[f"{dimension}_{criterion_name}"].item()
                    log[f"loss_val_{dimension}_{criterion_name}"] = loss_val[f"{dimension}_{criterion_name}"].item()
                
            hist["optimizer_loss"]["train"][epoch] = loss_train["optimizer_loss"].item()
            hist["optimizer_loss"]["val"][epoch] = loss_val["optimizer_loss"].item()

            log["loss_train_optimizer_loss"] = loss_train["optimizer_loss"].item()
            log["loss_val_optimizer_loss"] = loss_val["optimizer_loss"].item()

            hist["duration"][epoch] = epoch_duration

            if use_wandb:
                wandb.log(log)

            if (early_stop is not None and 
                early_stop(loss_val["optimizer_loss"])):
                break

        result = {}
        result["optimizer_loss_train"] = hist["optimizer_loss"]["train"]
        result["optimizer_loss_val"] = hist["optimizer_loss"]["val"]
        result["duration"] = hist["duration"]
        for dimension in self._criterions:
            for criterion_name in self._criterions[dimension]:
                result[f"{dimension}_{criterion_name}_train"] = hist[dimension][criterion_name]["train"]
                result[f"{dimension}_{criterion_name}_val"] = hist[dimension][criterion_name]["val"]

        return result

    




def print_info(loss_value:torch.Tensor, epoch:int, total_epochs:int, 
               time:float|None=None):
    """
    Prints the information of a epoch.

    Args:
        loss_value (torch.Tensor): epoch loss.
        epoch (int): epoch number.
        total_epochs (int): total number of epochs. 
        time (float, optional): time to run the epoch. Don't print if is 0.0. Defaults to 0.0.
        accuracy (float, optional): epoch accuracy.
    """

    print(f'Epoch [{epoch+1}/{total_epochs}], \
            Loss: {loss_value.item():.4f}', end="")

    if time is None:
        print("")
    else:
        print(f", Elapsed Time: {time:.2f} sec")
