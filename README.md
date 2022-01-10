

# Foundation Model Gym

纸上得来终觉浅，绝知此事要躬行。


## Step 0. [pytorch warmup](pytorch-warmup)

## Step 1. [Transformer](transformer) (1 week)

targets  
- a transformer implementation from scratch  

requirements  
- pytorch Transformer API (fine-grained)
- benchmarks (comparing with pytorch's implementation)  
- the Transformer paper

outputs  
- a transformer matching reported performances
- implementation notes/wiki

## Step 2. [core models](core-models) (1 week)

targets  
- the BERT implementation
- optimization skills (1 GPU + accumulated gradient)
- fine-tuning skills  

requirements  
- data pre-processing pipeline: tokenizers (Huggingface) + raw data --> pre-processed data --> MLM/NSP training data
- Huggenface BERT-base retrained with 10%, 20%, 30%, 40%, 50% training data, and their learning curves/MLM accuracies/NSP accuracies/GLUE performances
  * 1 GPU with small batch sizes or accumulated gradients
  * 8 GPUs with the official BERT setting
- GLUE evaluation toolkits
- the BERT paper

outputs  
- BERT base model matching reported performances on benchmarks
- implementation notes/wiki

## Step 3. [core peripherals](core-peripherals) (2 weeks)

targets  
- strategies on building vocabulary (tokenizers (word piece/sentence piece/bpe))
- strategies on positional embeddings
- masking, sampling, ...


requirements  
- raw data
- Huggingface tokenizer API
- data preprocessing papers

outputs  
- data pre-processing pipelines
- implementation notes/wiki


## Step 4. [fm with decoders](fm-with-decoders) (2 weeks)

targets  
- fm with decoders (GPT, unilm, BART, T5)  

requirements  
- raw data
- Hugging face API

outputs  
- fm models matching reported performances on benchmarks
- implementation notes/wiki


## Step 5. useful extensions (2 weeks)

more fm models  
- XLNet (different pre-training objectives) 
- tinyBERT, ALBERT (parameter sharing and compressing)
- roBERTa (data scale-up)

more pre-training, fine-tuning tricks  
- learning rates (layer-wise learning rates, warmup)
- training with resource constraints (early exit, accumulate gradient (batch size))

more data pre-processing  
- backdoor injection


## Reference


