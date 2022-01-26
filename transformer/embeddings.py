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
        pass

    def forward(
        self,
        input_ids: Tensor = None,
        token_type_ids: Tensor = None,
        position_ids: Tensor = None,
    ) -> Tensor:
        pass