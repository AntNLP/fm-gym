# script for Mask Language Model task
# corpus: wikipedia_en from https://huggingface.co/datasets/wikipedia
# wikipedia documents are segment by spacy (https://spacy.io)

# input_file format (json line): {"title": "...", "text": "..."}
# output_file format (json line): {"mask": ["...", ..., "..."], "label": {"pos": "true_token", ...}}

# python3.7.2 is required due to spacy

import json
import torch
import spacy
import random
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def torch_mask_tokens(input, tokenizer, mlm_probability=0.15):
    # copy from https://github.com/huggingface/transformers/blob/05fa1a7ac17bb7aa07b9e0c1e138ecb31a28bbfe/src/transformers/data/data_collator.py#L744
    # some modifications
    """
    Prepare masked tokens input/label for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    import torch

    label = input.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `mlm_probability`)
    probability_matrix = torch.full(label.shape, mlm_probability)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    label[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(label.shape, 0.8)).bool() & masked_indices
    input[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(label.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), label.shape, dtype=torch.long)
    input[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return input, label

def mask_text(input_file, output_file, mlm_probability=0.15, output_original_text=False, output_tokenized_text=False):
    input_file = open(input_file)
    output_file = open(output_file, 'w')

    for line in tqdm(input_file):
        data = json.loads(line)
        title = data["title"]
        text = data["text"]
        # filte \n \t \'
        text = text.replace('\n', '')
        text = text.replace('\t', '')
        text = text.replace('\'', "'")
        # segment document
        sentences = [sent.text for sent in nlp(text).sents]
        data = {}
        for sent in sentences:
            # TODO: support output_original_text & output_tokenized_text
            input_ids = tokenizer.encode(sent, add_special_tokens=False)
            mask_input_ids, mask_label = torch_mask_tokens(torch.LongTensor(input_ids), tokenizer, mlm_probability)
            mask_sents = tokenizer.convert_ids_to_tokens(mask_input_ids)
            mask_label = tokenizer.convert_ids_to_tokens(mask_label)
            mask_dict = dict(filter(lambda x: x[1] != '[UNK]', list(zip(range(len(mask_label)), mask_label))))
            data["masked"] = mask_sents
            data["label"] = mask_dict
            output_file.write(json.dumps(data, ensure_ascii=False) + '\n')

    input_file.close()
    output_file.close()

if __name__ == '__main__':

    set_seed(42)
    nlp = spacy.load('en_core_web_sm')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_file = '/mnt/data1/public/corpus/Bert_Pretrain/Raw_Wikipedia_EN/raw_wikipedia_en.json'
    output_file = '/mnt/data1/public/corpus/Bert_Pretrain/Raw_Wikipedia_EN/mlm_wikipedia_en.json'
    # WARNING: Token indices sequence length is longer than the specified maximum sequence length for this model (610 > 512). Running this sequence through the model will result in indexing errors
    # There may alert a warning: Token indices sequence length is longer than the specified maximum sequence length for this model
    # because some segmented sentences length longer that 512

    # This processing is slow
    # May need ~120 hours for wikipedia-en
    mask_text(input_file, output_file)
