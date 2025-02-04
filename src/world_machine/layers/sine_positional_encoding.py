import torch

class SinePositionalEncoding(torch.nn.Module):
    """
    Positional enconding using sine/cossine function.
    """
    def __init__(self, embed_dim:int, sequence_size:int) -> None:
        """
        Creates the layer.

        Args:
            embed_dim (int): embedding size in the input and output.
            sequence_size (int): size of the sequence in the input and output.
        """

        super().__init__()

        #Caches the positions encodings:
        position = torch.arange(sequence_size, dtype=torch.float32)
        expoent = 2.0*torch.arange(embed_dim, dtype=torch.float32)/embed_dim

        pe = torch.empty((sequence_size, embed_dim))

        pe.T[:] = position
        pe /= torch.pow(1e4, expoent)

        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])

        self.register_buffer("pe", pe)

    def forward(self) -> torch.Tensor:
        """
        Adds the positions encodings to the input.

        Args:
            input_tensor (torch.Tensor): input tensor to receive the positions encodings

        Returns:
            torch.Tensor: input + positional encoding.
        """
        return self.pe