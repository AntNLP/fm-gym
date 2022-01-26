from torch import Tensor


class LearnableEmbeddings(nn.Module):
    """Construct the Learnable embeddings."""
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        pad_token_id: int,
    ) -> None:
        pass

    def forward(
        self,
        ids: Tensor,
    ) -> Tensor:
        pass

class SinusoidalEmbeddings(nn.Module):
    """Construct the Sinusoidal embeddings."""
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        pad_token_id: int,
        div_scale: int = 10000,
    ) -> None:
        pass

    def forward(
        self,
        ids: Tensor,
    ) -> Tensor:
        pass


class BERTEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        pad_token_id: int = 0,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
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
        """
        pass

    def forward(
        self,
        input_ids: Tensor = None,
        token_type_ids: Tensor = None,
        position_ids: Tensor = None,
    ) -> Tensor:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
                Indices can be obtained using [`BertTokenizer`].
            token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
                1]`:
                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.max_position_embeddings - 1]`.
        """
        pass