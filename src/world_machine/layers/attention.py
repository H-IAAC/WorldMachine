import torch

from world_machine.profile import profile_range

from .positional_encoder import create_positional_encoder


class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(self, embed_dim: int, n_head: int, is_causal: bool, positional_encoder_type: str | None = None):
        super().__init__()

        self.attention = MultiHeadAttention(
            embed_dim, n_head, is_causal, positional_encoder_type)

    @profile_range("multi_head_self_attention_forward", domain="world_machine")
    def forward(self, x: torch.Tensor):
        return self.attention(x, x, x)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, n_head: int, is_causal: bool, positional_encoder_type: str | None = None) -> None:
        """
        Creates the layer.

        Args:
            embed_dim (int): size of the embedding in the layer input and output.
        """
        super().__init__()

        self.embed_dim = embed_dim

        self.n_head = n_head
        self.head_dim = embed_dim//n_head

        self.is_causal = is_causal

        if self.head_dim * n_head != embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads ({embed_dim}/{n_head} is not integer).")

        # Initialize weights

        # d_model = dv = dk = embed_dim
        # h = 1

        wQ = torch.Tensor(embed_dim, embed_dim)  # embed, embed
        wK = torch.Tensor(embed_dim, embed_dim)  # embed, dk
        wV = torch.Tensor(embed_dim, embed_dim)  # embed, dv
        w0 = torch.Tensor(embed_dim, embed_dim)  # embed, embed

        self.wQ = torch.nn.Parameter(wQ)
        self.wK = torch.nn.Parameter(wK)
        self.wV = torch.nn.Parameter(wV)
        self.w0 = torch.nn.Parameter(w0)

        self.register_buffer("dk_root", torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)))
        self.dk_root: torch.Tensor

        self._positional_encoder = create_positional_encoder(
            positional_encoder_type, embed_dim, 0, n_head)

        for w in [self.wQ, self.wK, self.wV, self.w0]:
            torch.nn.init.kaiming_normal_(w)

    @profile_range("multi_head_attention_forward", domain="world_machine")
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Process the inputs using the attention process.

        Input tensors must be in [batch, sentence, embed] order.

        Args:
            query (torch.Tensor): queries tensor, are compared against the keys.
            key (torch.Tensor): keys tensor, represents the keys.
            value (torch.Tensor): values tensor.

        Returns:
            torch.Tensor: the layer output, the values pondered by the compability between the keys and queries.
        """

        # Check input
        if query.shape[2] != self.embed_dim:
            raise ValueError(
                f"Inputs must have embed dimension of {self.embed_dim} ({query.shape[2]} != {self.embed_dim})")

        # Get dimensions
        batch_size = query.shape[0]
        context_size = query.shape[1]

        # Linear input transformation
        # Transpose weights because PyTorch does that
        with profile_range("linear_input_transformation", category="multi_head_attention", domain="world_machine"):
            Q = query @ self.wQ.T
            K = key @ self.wK.T
            V = value @ self.wV.T

        # batch_size, sentence, embed
        # to
        # batch_size,  n_head, sentence, head_dim
        with profile_range("pre_reshape", category="multi_head_attention", domain="world_machine"):
            Q = Q.transpose(0, 1).reshape(context_size, batch_size *
                                          self.n_head, self.head_dim).transpose(0, 1)
            K = K.transpose(0, 1).reshape(context_size, batch_size *
                                          self.n_head, self.head_dim).transpose(0, 1)
            V = V.transpose(0, 1).reshape(context_size, batch_size *
                                          self.n_head, self.head_dim).transpose(0, 1)
        # Now we have [
        # [batch0word0part0, batch0word1part0],
        # [batch0word0part1, batch0word1part1],
        # [batch1word0part0, batch1word1part0],
        # [batch1word0part1, batch1word1part1],
        # ]

        with profile_range("scores_computation", category="multi_head_attention", domain="world_machine"):
            scores = Q @ K.transpose(-2, -1)  # K.permute(0,1,3,2)
            scores /= self.dk_root

        # Apply causal bias
        with profile_range("causal_bias", category="multi_head_attention", domain="world_machine"):
            if self.is_causal:
                mask = torch.ones(
                    (context_size, context_size), dtype=torch.bool, device=query.device)
                mask = mask.tril()  # Lower triangular is one
                # Upper triangular without diagonal is ones
                mask = torch.bitwise_not(mask)

                attention_bias = torch.zeros(
                    (context_size, context_size), device=query.device)
                attention_bias[mask] = -torch.inf

                scores += attention_bias

        with profile_range("positional_encoder", category="multi_head_attention", domain="world_machine"):
            scores = self._positional_encoder.apply_attention_scores_pe(scores)

        probs = torch.softmax(scores, dim=-1)
        E = probs @ V

        # Return elements to correct place
        with profile_range("post_reshape", category="multi_head_attention", domain="world_machine"):
            E = E.reshape(batch_size, self.n_head, context_size, self.head_dim)
            E = E.transpose(-3, -2)
            E = E.reshape(batch_size, context_size, self.embed_dim)
        # Now we have [
        # [batch0word0, batch0word1],
        # [batch1word0, batch1word1]
        # ]

        result = E @ self.w0.T

        return result
