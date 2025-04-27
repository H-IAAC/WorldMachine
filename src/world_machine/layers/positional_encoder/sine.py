import torch

from .positional_encoder import PositionalEncoder


class SinePositionalEncoder(PositionalEncoder):
    """
    Positional enconding using sine/cossine function.
    """

    def __init__(self, embed_dim: int, sequence_size: int, n_head: int) -> None:
        """
        Creates the layer.

        Args:
            embed_dim (int): embedding size in the input and output.
            sequence_size (int): size of the sequence in the input and output.
        """

        super().__init__(embed_dim, sequence_size, n_head)

        # Caches the positions encodings:
        position = torch.arange(sequence_size, dtype=torch.float32)
        expoent = 2.0*torch.arange(embed_dim, dtype=torch.float32)/embed_dim

        pe = torch.empty((sequence_size, embed_dim))

        pe.T[:] = position
        pe /= torch.pow(1e4, expoent)

        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])

        self.register_buffer("pe", pe)
        self.pe: torch.Tensor

    def apply_input_pe(self, x: torch.Tensor) -> torch.Tensor:
        sequence_size = x.shape[1]

        return x+self.pe[:sequence_size, :]

    def remove_input_pe(self, x: torch.Tensor) -> torch.Tensor:
        sequence_size = x.shape[1]

        return x-self.pe[:, :sequence_size]
