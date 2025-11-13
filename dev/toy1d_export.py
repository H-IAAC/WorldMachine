
import pickle

from hamilton import driver
from world_machine_experiments import shared
from world_machine_experiments.toy1d import base

if __name__ == "__main__":

    base_args = {"sequence_length": 1000,
                 "n_sequence": 10000,
                 "context_size": 200,
                 "seed": [42, 0],
                 "output_dir": "toy1d_export"}

    d = driver.Builder().with_modules(base, shared).build()

    outputs = d.execute(["save_toy1d_datasets"], inputs=base_args)
