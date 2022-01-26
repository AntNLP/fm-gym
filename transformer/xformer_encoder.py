from typing import Union
from torch import Tensor


class XformerEncoder(nn.Module):
    """
    Construct the Transformer Encoder,
    consisting of a self-attention network and a feedforward network.
    """
    def __init__(
        self,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        attention_probs_dropout_prob: int = 0.1,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
        position_embedding_type: str = 'absolute',      # Supporting 'relative' is a bonus.
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
    ) -> None:
        """
        Args:
            num_hidden_layers (`int`, *optional*, defaults to 12):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (`int`, *optional*, defaults to 12):
                Number of attention heads for each attention layer in the Transformer encoder.
            hidden_size (`int`, *optional*, defaults to 768):
                Dimensionality of the encoder layers.
            attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
                The dropout ratio for the attention probabilities.
            layer_norm_eps (`float`, *optional*, defaults to 1e-12):
                The epsilon used by the layer normalization layers.
            hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
                The dropout probability for all fully connected layers in the embeddings and encoder.
            position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
                Type of position embedding. Choose one of `"absolute"`, `"relative"`. For
                positional embeddings use `"absolute"`. For more information on `"relative"`, please refer to
                [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            intermediate_size (`int`, *optional*, defaults to 3072):
                Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
                The non-linear activation function (function or string) in the encoder.
        """
        pass

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> tuple(Tensor, Tensor, Tensor):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Contextual representations.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
        """
        pass
