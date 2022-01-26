from torch import Tensor


class BertModel(nn.Module):
    """
    Construct the Bert model,
    consisting of a BERTEmbeddings network, a Transformer Encoder network, and two classifiers.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        pad_token_id: int = 0,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        attention_probs_dropout_prob: int = 0.1,
        position_embedding_type: str = 'absolute',
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        mlm_head_act: str = "gelu",
        nsp_head_act: str = "tanh",
        initializer_range: float = 0.02,
    ) -> None:
        """
        Args:
            vocab_size (`int`, *optional*, defaults to 30522):
                Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
                `inputs_ids`.
            hidden_size (`int`, *optional*, defaults to 768):
                Dimensionality of the encoder layers.
            max_position_embeddings (`int`, *optional*, defaults to 512):
                The maximum sequence length that this model might ever be used with. Typically set this to something large
                just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size (`int`, *optional*, defaults to 2):
                The vocabulary size of the `token_type_ids`.
            layer_norm_eps (`float`, *optional*, defaults to 1e-12):
                The epsilon used by the layer normalization layers.
            hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
                The dropout probability for all fully connected layers in the embeddings and encoder.
            num_hidden_layers (`int`, *optional*, defaults to 12):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (`int`, *optional*, defaults to 12):
                Number of attention heads for each attention layer in the Transformer encoder.
            attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
                The dropout ratio for the attention probabilities.
            position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
                Type of position embedding. Choose one of `"absolute"`, `"relative"`. For
                positional embeddings use `"absolute"`. For more information on `"relative"`, please refer to
                [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            intermediate_size (`int`, *optional*, defaults to 3072):
                Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
                The non-linear activation function (function or string) in the encoder.
            mlm_head_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
                The non-linear activation function (function or string) in the MLM Head.
            nsp_head_act (`str` or `Callable`, *optional*, defaults to `"tanh"`):
                The non-linear activation function (function or string) in the NSP Head.
            initializer_range (`float`, *optional*, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        """
        pass

    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None,
        token_type_ids: Tensor = None,
        position_ids: Tensor = None,
        mlm_labels: Tensor = None,      # if mlm_labels or nsp_labels: return loss of them
        nsp_labels: Tensor = None,
        output_attentions: bool = False,        # else: return (last_hid, attentions, hid_states)
        output_hidden_states: bool = False,
    ) -> tuple(Tensor, Tensor, Tensor):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
                Indices can be obtained using [`BertTokenizer`].
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
                1]`:
                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.max_position_embeddings - 1]`.
            mlm_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            nsp_labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:
                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
        """
        pass


