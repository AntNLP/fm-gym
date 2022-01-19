# data prepare

We use [wikipedia-en](https://huggingface.co/datasets/wikipedia) and [book-corpus](https://huggingface.co/datasets/bookcorpus)(not support yet) as pre-training dataset follow [BERT](https://arxiv.org/abs/1810.04805).

## step 1
`segment_documents.py` divides the document into one sentence per line, using blank lines to separate the documents for meeting input format of step 2.

This script has two arguments: output_file for path of output file and document_num for number of document need precessing.

Example:
```text
I am very happy.
Here is the second sentence.

A new document.
```

Usage:
```sh
output_file='/mnt/data1/public/corpus/Bert_Pretrain/Raw_Wikipedia_EN/segmented_wikipedia_en_100k.txt'
document_num=100000

python segment_documents.py \
    --output_file $output_file \
    --document_num $document_num
```

## step 2
`create_training_data.py` constructs pre-processed documents into data for training MLM and NSP tasks and is a implement using tensorflow. (huggingface version is not support yet)

Output format:
```jsonl
{"tokens": ["[CLS]", "...", "[SEP]", "...", "[SEP]"], "masked_positions": [], "masked_tokens": [], "next_sentence_label": 0}
{"tokens": ["[CLS]", "see", "also", "list", "of", "township", "-", "level", "divisions", "of", "tianjin", "##re", "##ference", "##s", "http", ":", "/", "/", "arts", ".", "cultural", "-", "china", ".", "com", "/", "en", "/", "[MASK]", "[MASK]", "[MASK]", "[MASK]", "##5", ".", "html", "##cate", "##gor", "##y", ":", "towns", "in", "tianjin", "[SEP]", "is", "a", "tonga", "##n", "boxer", ".", "he", "competed", "in", "the", "[MASK]", "'", "s", "welterweight", "event", "[MASK]", "the", "1984", "summer", "olympics", ".", "references", "##cate", "##gor", "##y", ":", "1963", "births", "##cate", "##gor", "##y", "[MASK]", "living", "[MASK]", "[MASK]", "##gor", "fired", ":", "welterweight", "[MASK]", "fortress", "[MASK]", "[MASK]", ":", "tonga", "##n", "male", "boxers", "##cate", "##gor", "##y", ":", "olympic", "boxers", "of", "tonga", "##cate", "##gor", "##y", ":", "boxers", "at", "the", "1984", "summer", "olympics", "##cate", "##gor", "##y", "[MASK]", "place", "of", "birth", "[MASK]", "(", "living", "people", ")", "[SEP]"], "masked_positions": [28, 29, 30, 31, 32, 53, 58, 74, 76, 77, 78, 79, 82, 83, 84, 85, 112, 116], "masked_tokens": ["65", "##arts", "##47", "##9", "##5", "men", "at", ":", "people", "##cate", "##gor", "##y", "boxers", "##cate", "##gor", "##y", ":", "missing"], "next_sentence_label": 1}
```

This format meets the needs of both MLM and NSP tasks.
[CLS] and [SEP] have already been added.
mask_positions indicates the locations that are masked off (from index 0 and token [CLS]), and mask_tokens indicates the correct tokens.
A next_sentence_label of 0 means second sentence is next sentence of first sentence.


Usage (same to `create_training_data.sh`):
```sh
input_file='/mnt/data1/public/corpus/Bert_Pretrain/Raw_Wikipedia_EN/segmented_wikipedia_en_100k.txt'
output_file='/mnt/data1/public/corpus/Bert_Pretrain/Raw_Wikipedia_EN/training_data_wikipedia_en_100k.jsonl'
vocab_file='./vocab.txt'

python create_pretraining_data.py \
    --input_file $input_file \
    --output_file $output_file \
    --vocab_file $vocab_file \
    --do_lower_case True \
    --do_whole_word_mask False \
    --max_seq_length 512 \
    --max_predictions_per_seq 77 \
    --random_seed 42 \
    --dupe_factor 1 \
    --short_seq_prob 0.1
```

`vocab.txt` should be consistent with `bert-base-uncased` and can be downloaded from [here](https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt).

Refer to `create_training_data.py` for detailed description of the parameters.
# references

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
2. https://github.com/google-research/bert
