from typing import List, Dict


class Tokenizer():
    """
    Constructing a tokenizer, including training and encoding.
    """
    def __init__(self,
                 vocab: Dict[str, int] = None,
                 unk_token: str = "[UNK]",
                 prefix: str = "##",
                 lowercase: bool = False,
                 **kwargs) -> None:
        """
        Args:
            vocab (`Dict[str, int]`, optional, defaults to `None`):
                A dictionnary of string keys and their ids `{"am": 0,...}`.
            unk_token (`str`, optional, defaults to `[UNK]`):
                The unknown token to be used by the model.
            prefix (`str`, optional, defaults to `##`):
                A prefix to be used for every subword that is not a beginning-of-word.
            lowercase (`bool`, optional, defaults to `False`):
                Whether to lowercase.
        """

        if vocab is None:
            self.vocab = {}
        else:
            self.vocab = vocab

        pass

    def train(self,
              files: List[str],
              vocab_size: int = 30000,
              min_frequency: int = 2,
              special_tokens: List[str] = [
                  "[PAD]",
                  "[UNK]",
                  "[CLS]",
                  "[SEP]",
                  "[MASK]",
              ],
              limit_alphabet: int = 1000,
              initial_alphabet: List[str] = [],
              prefix: str = "##",
              **kwargs) -> None:
        """Build vocabulary
        Args:
            files (`List[str]`):
                A list of path to the files that we should use for training.
            vocab_size (`int`, optional, default to `30000`):
                The size of the final vocabulary, including all tokens and alphabet. Note that 30000 for BPE and WordPiece, while 8000 for SentencePieceUnigram.
            min_frequency (`int`, optional, default to `2`):
                The minimum frequency a pair should have in order to be merged. Note that 0 for WordPiece and SentencePieceUnigram, while 2 for BPE.
            special_tokens (`List[str]`, optional, default to `["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",]`):
                A list of special tokens the model should know of.
            limit_alphabet (`int`, optional, default to `1000`):
                The maximum different characters to keep in the alphabet.
            initial_alphabet (`List[str]`, optional, default to `[]`)
                A list of characters to include in the initial alphabet, even if not seen in the training dataset. If the strings contain more than one character, only the first one is kept.
            prefix (`str`, optional, `##`):
                A prefix to be used for every subword that is not a beginning-of-word.
        """
        pass

    def encode(self, sequence: str) -> List[str]:
        """Tokenize
        Args:
            sequence (`str`):
                The raw text sequence we want to encode.
        """
        pass

    def save(self, path: str) -> None:
        """Save the class `Tokenizer` to the file at the given path.

        Args:
            path (`str`):
                A path to a file in which to save the serialized tokenizer.
        """
        pass

    def from_file(path: str) -> Tokenizer:
        """Instantiate a new class `Tokenizer` from the file at the given path.

        Args:
            path (`str`):
                A path to a local JSON file representing a previously serialized
                class `Tokenizer`.
        """
        pass
