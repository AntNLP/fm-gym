from torch import Tensor


class FeedforwardSublayer(nn.Module):
    """
    Construct the Feedforward Sublayer, consisting of two linear networks,
    a layer normalization network, and a residual network.
    """
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ) -> None:
        pass

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        pass

