import torch
from torch.utils.data import DataLoader


def toy1d_simple_shift_loss(toy1d_dataloaders: dict[str, DataLoader]) -> float:
    mse = torch.nn.MSELoss()

    total_loss = torch.tensor(0, dtype=torch.float32)
    n = 0

    with torch.no_grad():
        for item in toy1d_dataloaders["train"]:
            inputs: torch.Tensor = item["inputs"]
            targets: torch.Tensor = item["targets"]

            prev_values = inputs["state_decoded"][:, :, 0]
            next_values = targets["state_decoded"][:, :, 0]

            total_loss += mse(prev_values, next_values) * targets.size(0)

            n += targets.size(0)

        total_loss = total_loss.item()
        total_loss /= n

    return total_loss
