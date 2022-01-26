from typing import Union
from torch import Tensor


class XformerDecoder(nn.Module):
    """
    Construct the Transformer Decoder,
    consisting of a masked self-attention network, a cross-attention network, and a feedforward network.
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
        encoder_hidden_states: Tensor = None,
        encoder_attention_mask: Tensor = None,
        output_attentions: bool = False,        # Return Tensor if not output_attentions else tuple(Tensor, Tensor)
    ) -> Union[Tensor, tuple(Tensor, Tensor)]:
        pass
