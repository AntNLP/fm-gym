## Tokenizer

To build a tokenizer, we need to perform three steps:
+ pre-process: split words according to whitespace and punctuation, or using tools like [spaCy](https://spacy.io/) and [Moses](https://www.statmt.org/moses/?n=Development.GetStarted).
+ train: build vocabulary on the corpus.
+ encode: output sub-words according to the vocabulary.

### Resources & References
+ Paper:
  + [Byte Pair Encoding (BPE)](https://aclanthology.org/P16-1162.pdf)
  + [WordPiece](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)
  + [SentencePiece Unigram](https://arxiv.org/pdf/1804.10959.pdf)
+ Code:
  + [huggingface](https://github.com/huggingface/tokenizers) (Rust implementation)
  + [BPE](https://github.com/rsennrich/subword-nmt/tree/master/subword_nmt) (Python implementation)
  + [BPE (light version)](https://github.com/lovit/WordPieceModel/blob/master/wordpiecemodel/bpe.py) (Python implementation)
  + [SentencePiece](https://github.com/google/sentencepiece) (C++ implementation)
  + [WordPiece](https://github.com/google-research/bert/blob/master/tokenization.py) (Python Implementation, without training code)
+ blog:
  + https://huggingface.co/docs/transformers/tokenizer_summary
  + https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46
  + https://towardsdatascience.com/wordpiece-subword-based-tokenization-algorithm-1fbd14394ed7

### Implementation
For clarity, we assume that the corpus has been pre-processed with spaCy. Thus, the structure of [`Tokenizer` class](tokenizer.py) as follow:
+ `__init__`: initialize.
+ `train`: build the vocabulary.
+ `encode`: output sub-words according to the vocabulary.
+ `save`: save the class to the file.
+ `from_file`: instantiate a new class from the file.

### Verification
##### Our Output
```python
# Step 1: Set the random seed
SEED=xxx
random.seed(SEED)
np.random.seed(SEED)

# Step 2: Prepare the corpus
# one line per sentence, words are split by whitespace
corpus_file = "xxx"

# Step 3: Build the vocabulary
tokenizer = Tokenizer(
    vocab=None,
    unk_token="[UNK]"
    ...
)
tokenizer.train(
    files=[corpus_file],
    vocab_size=30000,
    ...
)
tokenizer.save("tokenizer.json")

# Step 4: Tokenize
tokenizer = Tokenizer.from_file("tokenizer.json")
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
```


##### Huggingface's Output
```python
# Step 1: Set the random seed
# IMPORTANT! SEED must be the same as ours!
SEED=xxx
random.seed(SEED)
np.random.seed(SEED)

# Step 2: Prepare the corpus
# one line per sentence, words are split by whitespace
corpus_file = "xxx"

# Step 3: Build the vocabulary, here taking BPE as an example
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# We should keep the same hyper-parameters with ours
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train([corpus_file], trainer)

tokenizer.save("tokenizer.json")

# Step 4: Tokenize
tokenizer = Tokenizer.from_file("tokenizer.json")
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
```

Lastly, to verify the correctness of our implementation, we should compare huggingface's output with ours.
