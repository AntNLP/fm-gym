## Data pre-processing pipeline

In this section, we need to pre-process raw text content to pre-training data and then store them locally for persistency.

We use [wikipedia-en](https://huggingface.co/datasets/wikipedia) as a pre-training dataset and refer the data pre-processing pipeline following BERT.

### Step 1. segment documents
The raw data is formatted as `"title": "...", "text": "..."` JSON format (stored at [here](172.20.3.63:/mnt/data1/public/corpus/Bert_Pretrain/Raw_Wikipedia_EN/raw_wikipedia_en.json), right click to copy). Firstly, we segment all documents into sentences (one line one sentence) and split these segmented documents by one blank line (using [spaCy](https://spacy.io/) or other NLP toolkit). 

### Step 2. creating pre-training data
Then, we make maksed pre-training data using segmented documents by the masking policies of BERT (refer to [this script](https://github.com/google-research/bert/blob/master/create_pretraining_data.py) of original implementation of google or [the DataCollatorForLanguageModeling class](https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForLanguageModeling) of Hugging Face). The data format should be meet the requirement of the MLM ans NSP tasks as `{"tokens": ["[CLS]", "...", "[SEP]", "...", "[SEP]"], "masked_positions": [], "masked_tokens": [], "next_sentence_label": 0}`. We save pre-processed data locally for persistency (you may fixed the random seed for reproduction).

The `tokens` is the tokens of tokenized sentence. The `masked_positions` and `masked_tokens` is created for the MLM task that reveal which positions of tokens are masked and their real tokens (only special token `[MASK]` can't infer that information because mask policies of BERT will remain some chosen tokens unchanged or replace them with random tokens). The field of `next_sentence_label` is using for NSP task.

### Step 3. change data format
To reduce disk space consumption, we use compact JSON fields. However, in order to satisfy the model inputs, we also need a script to process the persistent data into the inputs needed by the model. The detailed input and output can be see [here](./pretraining_data_analyzer.py).

You need to complete the following classes for data pro-precessing:

- [Segmentor](./segmentor.py)
- [PreTrainingDataMaker](./pretraining_data_maker.py)
- [PreTrainingDataAnalyzer](./pretraining_data_analyzer.py)
- [PreTrainingDataset](./pretraining_dataset.py)

The second class struct refers to the original google implementation. The detailed definition of the function can be found at [here](https://github.com/google-research/bert/blob/master/create_pretraining_data.py).