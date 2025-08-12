import torch

from world_machine.profile import profile_range


class PointwiseFeedforward(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_size: int,
                 dropout_rate: float = 0.0, output_dim: int | None = None):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        self.linear1 = torch.nn.Linear(input_dim, hidden_size)
        self.dropout_linear1 = torch.nn.Dropout(dropout_rate)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, output_dim)
        self.dropout_linear2 = torch.nn.Dropout(dropout_rate)

    @profile_range("pointwiseff_forward", domain="world_machine")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout_linear1(self.linear1(x))
        y = self.relu(y)
        y = self.dropout_linear2(self.linear2(y))

        return y
