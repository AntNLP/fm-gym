To implement a correct **Transformer Encoder-Decoder** model, we can refer to some resources:

0. Transformer paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
1. [Pytorch Implementation](https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer)
2. HuggingFace Implementation: [BERT](https://github.com/huggingface/transformers/tree/master/src/transformers/models/bert) for Encoder & [BART](https://github.com/huggingface/transformers/tree/master/src/transformers/models/bart) for Enc-Dec

In this project, we give the basic definition of the Transformer network (specifying the required classes and the calling logic between them).
**You can freely modify the definition with knowledge of the context.**
To build a generic Transformer we need to implement the following classes:
+ [XformerEncoder](https://github.com/AntNLP/fm-gym/blob/b9b0fee45616a52578c1e599073568e11b058f14/transformer/xformer_encoder.py#L5)
+ [XformerDecoder](https://github.com/AntNLP/fm-gym/blob/b9b0fee45616a52578c1e599073568e11b058f14/transformer/xformer_decoder.py#L5)
+ [SelfAttentionSublayer](https://github.com/AntNLP/fm-gym/blob/b9b0fee45616a52578c1e599073568e11b058f14/transformer/attention.py#L31)
+ [CrossAttentionSublayer](https://github.com/AntNLP/fm-gym/blob/b9b0fee45616a52578c1e599073568e11b058f14/transformer/attention.py#L59)
+ [MultiHeadAttention](https://github.com/AntNLP/fm-gym/blob/b9b0fee45616a52578c1e599073568e11b058f14/transformer/attention.py#L5)
+ [FeedforwardSublayer](https://github.com/AntNLP/fm-gym/blob/b9b0fee45616a52578c1e599073568e11b058f14/transformer/feedforward.py#L4)
+ [SinusoidalEmbeddings](https://github.com/AntNLP/fm-gym/blob/b9b0fee45616a52578c1e599073568e11b058f14/transformer/embeddings.py#L4)
+ [LearnableEmbeddings](https://github.com/AntNLP/fm-gym/blob/b9b0fee45616a52578c1e599073568e11b058f14/transformer/embeddings.py#L20)

The tree logic between classes is as follows:
```
├── XformerEncoder
│   ├── SelfAttentionSublayer
│   │   ├── MultiHeadAttention
│   ├── FeedforwardSublayer
├── XformerDecoder
│   ├── SelfAttentionSublayer
│   │   ├── MultiHeadAttention
│   ├── CrossAttentionSublayer
│   │   ├── MultiHeadAttention
│   ├── FeedforwardSublayer
├── LearnableEmbeddings
├── SinusoidalEmbeddings
```
To obtain the output of a reproduced Transformer, we can define a randomly initialized BERT (Encoder) or BART (Enc-Dec).
Take BERT as an example:
```
├── BertModel
│   ├── BERTEmbeddings
│   │   ├── LearnableEmbeddings
│   │   └── SinusoidalEmbeddings
│   ├── XformerEncoder
│   │   ├── SelfAttentionSublayer
│   │   │   ├── MultiHeadAttention
│   │   ├── FeedforwardSublayer
│   ├── BERTMLMHead (Optional)
│   ├── BERTNSPHead (Optional)
```
```python
# Step 1: Set the Random Seed in the program entry 
# earlier is better
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Step 2: Load BERT hyperparameters
from transformers import AutoTokenizer, AutoConfig
config = AutoConfig.from_pretrained(os.environ['TRANSFORMERS_CACHE']+'bert-base')   # 'bart-base'

# Step 3: Define our BERT
from bert_model import BertModel
our_bert = BertModel(
    vocab_size = config.vocab_size,
    hidden_size = config.hidden_size,
    ...
    ...
)

# Step 4: Get the model output without dropout
our_bert.eval()
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
last_hidden_states, _, _ = our_bert(**inputs)
```

By referring to the hugging face implementation, we could confirm that we have implemented the **Transformer Encoder-Decoder** model correctly.
Check that our output is consistent with the BERT/BART output ( `last_hidden_states` in the code below).

```python
# Step 1: Set the Random Seed in the program entry 
# IMPORTANT! SEED must be the same as ours!
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Step 2: Define a randomly initialized BART model
from transformers import AutoTokenizer, AutoConfig, AutoModel

config = AutoConfig.from_pretrained(os.environ['TRANSFORMERS_CACHE']+'bert-base')   # 'bart-base'
model = AutoModel.from_config(config)
tokenizer = AutoTokenizer.from_pretrained(os.environ['TRANSFORMERS_CACHE']+'bert-base')  # 'bart-base'

# Step 3: Get the model output without dropout
model.eval()
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
```

