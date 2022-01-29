class TrainingInstance:
    """A single training Instance (sentence pair)."""

    def __init__(
        self,
        tokens,
        token_type_ids,
        masked_positions,
        masked_labels,
        next_sentence_label,
    ):
        self.tokens = tokens
        self.token_type_ids = token_type_ids
        self.next_sentence_label = next_sentence_label
        self.masked_positions = masked_positions
        self.masked_labels = masked_labels


class PreTrainingDataMaker:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        tokenizer,
        max_seq_length: int = 512,
        random_seed: int = 42,
        mlm_prob: float = 0.15,
        short_seq_prob: float = 0.1,
    ):
        """The class is pre-training data maker for the MLM ans NSP tasks.

        Parameters
        ----------
        input_file : str
            "Input raw text file."
        output_file : str
            "Output TF example file."
        max_seq_length : int, optional
            "Maximum sequence length.", by default 512
        random_seed : int, optional
            "Random seed for data generation.", by default 42
        mlm_prob : float, optional
            "Masked LM probability.", by default 0.15
        short_seq_prob : float, optional
            "Probability of creating sequences which are shorter than the " "maximum length.", by default 0.1
        """
        pass

    def write_instance_to_output_file(self, instances):
        """Output JSON file from `TrainingInstance`s.
        """
        pass

    def create_training_instances(self):
        """Create `TrainingInstance`s from raw text.
        """
        all_documents = [[]]

        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. ocument boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        pass
        # return instances

    def create_instances_from_document(self, all_documents, document_index):
        """Creates `TrainingInstance`s for a single document.
        """
        pass
        # return instances

    def create_masked_lm_predictions(self, tokens, masked_prob):
        """Creates the predictions for the masked LM objective.
        """
        pass
        # return (output_tokens, masked_positions, masked_labels)

    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens):
        """Truncates a pair of sequences to a maximum sequence length.
        """
        pass
