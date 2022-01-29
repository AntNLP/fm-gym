from torch.utils.data import Dataset

class PreTrainingDataset(Dataset):
    def __init__(
        self, input_ids, token_type_ids, attention_mask, labels, next_sentence_label
    ):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.next_sentence_label = next_sentence_label

    def __len__(self):
        return len(self.next_sentence_label)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        token_type_ids = self.token_type_ids[idx]
        attention_mask = self.attention_mask[idx]
        labels = self.labels[idx]
        next_sentence_label = self.next_sentence_label[idx]
        return input_ids, token_type_ids, attention_mask, labels, next_sentence_label
