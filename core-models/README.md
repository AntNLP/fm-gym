# Pre-training data for MLM and NSP

We use [wikipedia-en](https://huggingface.co/datasets/wikipedia) and [book-corpus](https://huggingface.co/datasets/bookcorpus)(not support yet) as pre-training dataset follow [BERT](https://arxiv.org/abs/1810.04805).

## Step 1. segment documents
`segment_documents.py` divides the document into one sentence per line, using blank lines to separate the documents for meeting input format of step 2.

This script has two arguments: `output_file` for the path of the output file and `document_num` for the number of documents in the front that needs precessing.

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

## Step 2. create training data

There are two version of script for creating pre-training data: `create_pretraining_data_hf.py` and `create_pretraining_data_tf.py` using transformers library (hugging face) and tensorflow library respectively. `create_pretraining_data_hf.py` is recommended.

### Hugging Face Version (Recommended)

`create_pretraining_data_hf.py` constructs pre-processed documents in step 1 into data for training MLM and NSP tasks.

Output format:
```jsonl
{"tokens": ["[CLS]", "...", "[SEP]", "...", "[SEP]"], "masked_positions": [], "masked_tokens": [], "next_sentence_label": 0}
{"tokens": ["[CLS]", "see", "also", "list", "of", "township", "-", "level", "divisions", "of", "tianjin", "##re", "##ference", "##s", "http", ":", "/", "/", "arts", ".", "cultural", "-", "china", ".", "com", "/", "en", "/", "[MASK]", "[MASK]", "[MASK]", "[MASK]", "##5", ".", "html", "##cate", "##gor", "##y", ":", "towns", "in", "tianjin", "[SEP]", "is", "a", "tonga", "##n", "boxer", ".", "he", "competed", "in", "the", "[MASK]", "'", "s", "welterweight", "event", "[MASK]", "the", "1984", "summer", "olympics", ".", "references", "##cate", "##gor", "##y", ":", "1963", "births", "##cate", "##gor", "##y", "[MASK]", "living", "[MASK]", "[MASK]", "##gor", "fired", ":", "welterweight", "[MASK]", "fortress", "[MASK]", "[MASK]", ":", "tonga", "##n", "male", "boxers", "##cate", "##gor", "##y", ":", "olympic", "boxers", "of", "tonga", "##cate", "##gor", "##y", ":", "boxers", "at", "the", "1984", "summer", "olympics", "##cate", "##gor", "##y", "[MASK]", "place", "of", "birth", "[MASK]", "(", "living", "people", ")", "[SEP]"], "masked_positions": [28, 29, 30, 31, 32, 53, 58, 74, 76, 77, 78, 79, 82, 83, 84, 85, 112, 116], "masked_tokens": ["65", "##arts", "##47", "##9", "##5", "men", "at", ":", "people", "##cate", "##gor", "##y", "boxers", "##cate", "##gor", "##y", ":", "missing"], "next_sentence_label": 1}
```

This format meets the needs of both MLM and NSP tasks.
Special tokens of [CLS] and [SEP] have already been added.
`mask_positions` indicates the locations that are masked off (from index 0 and token [CLS]), and `mask_tokens` indicates the correct tokens.
**A next_sentence_label of 0 means second sentence is next sentence of first sentence.**

Usage:
```sh
segmented_file_path='/mnt/data1/public/corpus/Bert_Pretrain/Raw_Wikipedia_EN/segmented_wikipedia_en_100k.txt'
output_file_path='/mnt/data1/public/corpus/Bert_Pretrain/Raw_Wikipedia_EN/training_data_wikipedia_en_100k.jsonl'

python hf4mlm.py \
	--segmented_file_path $segmented_file_path
	--output_file_path $output_file_path 
	--model_name bert-base-uncased \
	--block_size 512 \
	--batch_size 32
```

### Tensorflow Version

`create_pretraining_data_tf.py` constructs pre-processed documents in step 1 into data for training MLM and NSP tasks.

The input and output are same with `create_pretraining_data_hf.py`

TF version script require a `vocab.txt` that should be consistent with `bert-base-uncased` and can be downloaded from [here](https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt).

The full usage of them can be see at `create_pretraining_data_hf.sh` and `create_pretraining_data_tf.sh`

Reading code for detailed description of the parameters.

# Resources
All data are stored at 172.20.3.63:/mnt/data1/public/corpus/Bert_Pretrain/Raw_Wikipedia_EN/

`raw_wikipedia_en.json` contains 6,078,422 documents of wikipedia-en as json format of {"title": "...", "text": "..."}

`mlm_wikipedia_en.json` contains 126,966,355 masked sentences line by line as json format of {"masked": "...", "label": {postion: real_token}}

`segmented_wikipedia_en_100k.txt` contains the segmented results of the first 100,000 wikipedia-en documents, one sentence per line, with blank line separating the documents.

`segmented_wikipedia_en_1m.txt` contains the segmented results of the first 1,000,000 wikipedia-en documents, one sentence per line, with blank line separating the documents.

`training_data_wikipedia_en_100k.jsonl` contains 301,550 training data that constructed by `create_pretraining_data.py` using `segmented_wikipedia_en_100k.txt`. The data format is {"tokens": ["[CLS]", "...", "[SEP]", "...", "[SEP]"], masked_positions: [...], masked_tokens: ["..."], "next_sentence_label": ...}

`training_data_wikipedia_en_1m.jsonl` contains 3,023,962 training data that constructed by `create_pretraining_data.py` using `segmented_wikipedia_en_1m.txt`. The data format is {"tokens": ["[CLS]", "...", "[SEP]", "...", "[SEP]"], masked_positions: [...], masked_tokens: ["..."], "next_sentence_label": ...}

# References

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
2. https://github.com/google-research/bert
