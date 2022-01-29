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
        """
        Args:
            num_attention_heads (`int`, *optional*, defaults to 12):
                Number of attention heads for each attention layer in the Transformer encoder.
            hidden_size (`int`, *optional*, defaults to 768):
                Dimensionality of the encoder layers.
            attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
                The dropout ratio for the attention probabilities.
            position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
                Type of position embedding. Choose one of `"absolute"`, `"relative"`. For
                positional embeddings use `"absolute"`. For more information on `"relative"`, please refer to
                [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
        """
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
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Contextual representations.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token & future token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Contextual representations of the encoding sequence.
            encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
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
        """
        Args:
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
        """
        assert position_embedding_type == 'absolute' or position_embedding_type =='relative'
        pass

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        output_attentions: bool = False,        # Return Tensor if not output_attentions else tuple(Tensor, Tensor)
    ) -> Union[Tensor, tuple(Tensor, Tensor)]:
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
        """
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
        """
        Args:
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
        """
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
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Contextual representations.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token & future token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Contextual representations of the encoding sequence.
            encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        pass


