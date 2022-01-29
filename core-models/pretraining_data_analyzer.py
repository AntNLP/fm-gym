class PreTrainingDataAnalyzer():
    def __init__(self, tokenizer):
        pass

    def analysis(self, json_data):
        """
        Parameters
        ----------
        json_data : dict
        {
            "tokens": ["[CLS]", "...", "[SEP]", "...", "[SEP]"], 
            "masked_positions": [], 
            "masked_tokens": [], 
            "next_sentence_label": 0 or 1
        }
        """
        pass
        # return (input_ids, token_type_ids, attention_mask, labels, next_sentence_label)
