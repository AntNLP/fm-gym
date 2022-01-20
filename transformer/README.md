To implement a correct **Transformer Encoder-Decoder** model, we can refer to two resources:

+ [Pytorch Implementation](https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer)
+ [HuggingFace Implementation: BART (highly recommended)](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bart/modeling_bart.py#L846)
  + [BartModel](https://github.com/huggingface/transformers/blob/e57468b8a85bad5cc17efbfcfdd3eecb9b8a62ec/src/transformers/models/bart/modeling_bart.py#L1123)
  + [BartEncoder](https://github.com/huggingface/transformers/blob/e57468b8a85bad5cc17efbfcfdd3eecb9b8a62ec/src/transformers/models/bart/modeling_bart.py#L671)
  + [BartDecoder](https://github.com/huggingface/transformers/blob/e57468b8a85bad5cc17efbfcfdd3eecb9b8a62ec/src/transformers/models/bart/modeling_bart.py#L846)
  + [BartLearnedPositionalEmbedding](https://github.com/huggingface/transformers/blob/e57468b8a85bad5cc17efbfcfdd3eecb9b8a62ec/src/transformers/models/bart/modeling_bart.py#L107)
  + [BartAttention](https://github.com/huggingface/transformers/blob/e57468b8a85bad5cc17efbfcfdd3eecb9b8a62ec/src/transformers/models/bart/modeling_bart.py#L127)



By referring to the hugging face implementation, we could confirm that we have implemented the **Transformer Encoder-Decoder** model correctly.
Check that our output is consistent with the Bart output ( `last_hidden_states` in the code below).

```python
# Step 1: Set the Random Seed in the program entry 
# IMPORTANT! SEED must be the same as ours!
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Step 2: Define a randomly initialized BART model
from transformers import AutoTokenizer, AutoConfig, AutoModel

config = AutoConfig.from_pretrained(os.environ['TRANSFORMERS_CACHE']+'bart-base')
model = AutoModel.from_config(config)
tokenizer = AutoTokenizer.from_pretrained(os.environ['TRANSFORMERS_CACHE']+'bart-base')

# Step 3: Get the model output without dropout
model.eval()
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
```

