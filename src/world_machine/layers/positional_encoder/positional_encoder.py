import torch


class PositionalEncoder(torch.nn.Module):
    def __init__(self, embed_dim: int, sequence_size: int, n_head: int):
        super().__init__()

        self._embed_dim = embed_dim
        self._sequence_size = sequence_size
        self._n_head = n_head

    def apply_input_pe(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def apply_attention_scores_pe(self, scores: torch.Tensor) -> torch.Tensor:
        return scores

    def remove_input_pe(self, x: torch.Tensor) -> torch.Tensor:
        return x
