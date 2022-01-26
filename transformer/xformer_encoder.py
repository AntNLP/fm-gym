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
        pass

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> tuple(Tensor, Tensor, Tensor):
        pass
