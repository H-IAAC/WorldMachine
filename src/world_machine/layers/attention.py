import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from world_machine.layers.positional_encoder import create_positional_encoder
from world_machine.profile import profile_range


def apply_score_mod(score, score_mode, batch_index, head_index, query_index, key_index):
    return score+score_mode[batch_index, head_index, query_index, key_index]


class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(self, embed_dim: int, n_head: int, is_causal: bool, positional_encoder_type: str | None = None, fast: bool = True):
        super().__init__()

        self.attention = MultiHeadAttention(
            embed_dim, n_head, is_causal, positional_encoder_type, fast, True)

    def pre_compute_attention_bias(self, size: int) -> None:
        self.attention.pre_compute_attention_bias(size)

    @profile_range("multi_head_self_attention_forward", domain="world_machine")
    def forward(self, x: torch.Tensor):
        return self.attention(x, x, x)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, n_head: int, is_causal: bool, positional_encoder_type: str | None = None, fast: bool = True, self_attention: bool = False) -> None:
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
        self.fast = fast

        if self.head_dim * n_head != embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads ({embed_dim}/{n_head} is not integer).")

        # Initialize weights

        # d_model = dv = dk = embed_dim
        # h = 1
        w0 = torch.Tensor(embed_dim, embed_dim)  # embed, embed

        self.w0 = torch.nn.Parameter(w0)

        self.register_buffer("dk_root", torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)))
        self.dk_root: torch.Tensor

        self._positional_encoder = create_positional_encoder(
            positional_encoder_type, embed_dim, 0, n_head)

        for w in [self.w0]:
            torch.nn.init.kaiming_normal_(w)

        self._self_attention = self_attention
        if self_attention:
            self.input_projection = torch.nn.Linear(
                embed_dim, 3*embed_dim, bias=False)
        else:
            self.input_projection_weights = torch.nn.Parameter(
                torch.empty((3 * embed_dim, embed_dim)))
            torch.nn.init.kaiming_normal_(self.input_projection_weights)

        self.register_buffer("attention_bias", torch.tensor([]), False)
        self.attention_bias: torch.Tensor

        self.local_only = False

    @profile_range("pre_compute_attention_bias", category="multi_head_attention", domain="world_machine")
    def pre_compute_attention_bias(self, size: int) -> None:
        self._compute_attention_bias(size)

    @profile_range("compute_attention_bias", category="multi_head_attention", domain="world_machine")
    def _compute_attention_bias(self, size: int) -> None:
        if self.attention_bias.dim() != 3 or self.attention_bias.shape[1] < size:

            attention_bias = torch.zeros(
                (size, size), device=self.w0.device)

            if self.is_causal:
                with profile_range("causal_bias", category="multi_head_attention", domain="world_machine"):

                    mask = torch.ones(
                        (size, size), dtype=torch.bool, device=self.w0.device)
                    mask = mask.tril()  # Lower triangular is one
                    # Upper triangular without diagonal is ones
                    mask = torch.bitwise_not(mask)

                    attention_bias[mask] = -torch.inf

            attention_bias = attention_bias.unsqueeze(
                0).repeat([self.n_head, 1, 1])

            with profile_range("positional_encoder", category="multi_head_attention", domain="world_machine"):
                attention_bias = self._positional_encoder.apply_attention_bias_pe(
                    attention_bias)

            self.attention_bias = attention_bias

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
        if self._self_attention and not (query is key and key is value):
            raise ValueError(
                "query, key and value must be the same in self attention")

        # Get dimensions
        batch_size = query.shape[0]
        context_size = query.shape[1]

        # Linear input transformation
        # Transpose weights because PyTorch does that
        with profile_range("linear_input_transformation", category="multi_head_attention", domain="world_machine"):
            # Q = query @ self.wQ.T
            # K = key @ self.wK.T
            # V = value @ self.wV.T

            if self._self_attention:
                Q, K, V = self.input_projection(
                    query).split(self.embed_dim, dim=2)
            else:
                Q, K, V = torch.nn.functional._in_projection_packed(
                    query, key, value, self.input_projection_weights, None)

        if self.local_only:
            E = V
        else:
            # Compute bias
            with profile_range("prepare_attention_bias", category="multi_head_attention", domain="world_machine"):
                self._compute_attention_bias(context_size)
                attention_bias = self.attention_bias[:, :context_size,
                                                     :context_size]

            # attention bias: [head*batch, context, context]

            if self.fast:
                E = self._fast_attention(Q, K, V, attention_bias)
            else:
                E = self._manual_attention(Q, K, V, attention_bias)

        result = E @ self.w0.T
        return result

    def _fast_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attention_bias: torch.Tensor) -> torch.Tensor:
        batch_size = Q.shape[0]
        context_size = Q.shape[1]
        embed_size = Q.shape[2]

        with profile_range("pre_reshape", category="multi_head_attention", domain="world_machine"):
            Q = Q.view(batch_size, -1, self.n_head,
                       self.head_dim).transpose(1, 2)
            K = K.view(batch_size, -1, self.n_head,
                       self.head_dim).transpose(1, 2)
            V = V.view(batch_size, -1, self.n_head,
                       self.head_dim).transpose(1, 2)

            # attention_bias: [head*batch, seq, seq]
            # attention_bias2: [bath, head, seq, seq]
            attention_bias = attention_bias.reshape(
                [1, self.n_head, context_size, context_size])

        with profile_range("scaled_dot_product_attention", category="multi_head_attention", domain="world_machine"):
            E = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, attn_mask=attention_bias, scale=1/self.dk_root)

        with profile_range("post_reshape", category="multi_head_attention", domain="world_machine"):
            E = E.transpose(1, 2).view(
                batch_size, context_size, embed_size)

        return E

    def _manual_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attention_bias: torch.Tensor) -> torch.Tensor:
        batch_size = Q.shape[0]
        context_size = Q.shape[1]

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

            attention_bias = attention_bias.repeat(batch_size, 1, 1)
        # Now we have [
        # [batch0word0part0, batch0word1part0],
        # [batch0word0part1, batch0word1part1],
        # [batch1word0part0, batch1word1part0],
        # [batch1word0part1, batch1word1part1],
        # ]

        with profile_range("scores_computation", category="multi_head_attention", domain="world_machine"):
            scores = Q @ K.transpose(-2, -1)  # K.permute(0,1,3,2)
            scores /= self.dk_root

        with profile_range("add_attention_bias", category="multi_head_attention", domain="world_machine"):
            print(scores.shape, attention_bias.shape)
            scores += attention_bias

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

        return E

    def __setstate__(self, state):
        super().__setstate__(state)

        if "local_only" not in state:
            self.local_only = False
