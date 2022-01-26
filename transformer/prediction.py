from torch import Tensor


class BERTMLMHead(nn.Module):
    """
    Construct the BERT Masked Language Model Prediction Head:
    Linear -> LayerNorm -> Linear
    """
    def __init__(
        self,
        hidden_size: int = 768,
        mlm_head_act: str = "gelu",
        vocab_size: int = 30522,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        pass

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        pass


class BERTNSPHead(nn.Module):
    """
    Construct the BERT Next Sentence Prediction Head:
    Linear -> Linear
    """

    def __init__(
        self,
        hidden_size: int = 768,
        nsp_head_act: str = "tanh",
        label_size: int = 2,
    ) -> None:
        pass

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        pass


