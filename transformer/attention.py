from typing import Union
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """Construct the Multi Head Attention."""
    def __init__(
        self,
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        attention_probs_dropout_prob: int = 0.1,
        position_embedding_type: str = 'absolute',      # Supporting 'relative' is a bonus.
    ) -> None:

        assert position_embedding_type == 'absolute' or position_embedding_type =='relative'
        pass

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        encoder_hidden_states: Tensor = None,
        encoder_attention_mask: Tensor = None,
        output_attentions: bool = False,        # Return Tensor if not output_attentions else tuple(Tensor, Tensor)
    ) -> Union[Tensor, tuple(Tensor, Tensor)]:

        self.multi_head_attention_type = 'cross-attention' if encoder_hidden_states else 'self-attention'
        pass


class SelfAttentionSublayer(nn.Module):
    """
    Construct the Self-Attention Sublayer, consisting of an attention network,
    a layer normalization network, and a residual network.
    It is a part of the Transformer Encoder.
    """
    def __init__(
        self,
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        attention_probs_dropout_prob: int = 0.1,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
        position_embedding_type: str = 'absolute',      # Supporting 'relative' is a bonus.
    ) -> None:

        assert position_embedding_type == 'absolute' or position_embedding_type =='relative'
        pass

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        output_attentions: bool = False,        # Return Tensor if not output_attentions else tuple(Tensor, Tensor)
    ) -> Union[Tensor, tuple(Tensor, Tensor)]:
        pass


class CrossAttentionSublayer(nn.Module):
    """
    Construct the Cross-Attention Sublayer, consisting of an attention network,
    a layer normalization network, and a residual network.
    It is a part of the Transformer Decoder.
    """
    def __init__(
        self,
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        attention_probs_dropout_prob: int = 0.1,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
        position_embedding_type: str = 'absolute',      # Supporting 'relative' is a bonus.
    ) -> None:

        assert position_embedding_type == 'absolute' or position_embedding_type =='relative'
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


