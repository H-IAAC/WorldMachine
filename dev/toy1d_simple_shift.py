import torch
from hamilton import driver
from hamilton_sdk import adapters
from torch.optim import Adam, AdamW
from world_machine_experiments import shared, toy1d
from world_machine_experiments.toy1d import (
    Channels, base, multiple, parameter_variation)

if __name__ == "__main__":
    tracker = adapters.HamiltonTracker(
        project_id=1,
        username="EltonCN",
        dag_name="toy1d_base"
    )

    d = driver.Builder().with_modules(base, shared).with_adapter(tracker).build()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

inputs = {"sequence_length": 1000,
          "n_sequence": 10000,
          "context_size": 200,
          "batch_size": 32,
          "seed": 42
          }

outputs = d.execute(["toy1d_simple_shift_loss"], inputs=inputs)

print("Simple Shift MSE: ", outputs["toy1d_simple_shift_loss"])
