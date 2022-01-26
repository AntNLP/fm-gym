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
        """
        Args:
            hidden_size (`int`, *optional*, defaults to 768):
                Dimensionality of the encoder layers.
            intermediate_size (`int`, *optional*, defaults to 3072):
                Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
                The non-linear activation function (function or string) in the encoder.
            layer_norm_eps (`float`, *optional*, defaults to 1e-12):
                The epsilon used by the layer normalization layers.
            hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
                The dropout probability for all fully connected layers in the embeddings and encoder.
        """
        pass

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Contextual representations.
        """
        pass

