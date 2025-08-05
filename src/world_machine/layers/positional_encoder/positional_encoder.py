import torch


class PositionalEncoder(torch.nn.Module):
    def __init__(self, embed_dim: int, sequence_size: int, n_head: int):
        super().__init__()

        self._embed_dim = embed_dim
        self._sequence_size = sequence_size
        self._n_head = n_head

    def apply_input_pe(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def apply_attention_bias_pe(self, attention_bias: torch.Tensor) -> torch.Tensor:
        return attention_bias

    def remove_input_pe(self, x: torch.Tensor) -> torch.Tensor:
        return x
